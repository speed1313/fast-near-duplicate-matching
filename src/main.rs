use clap::Parser;
use env_logger;
use near_duplicate_matching::{has_doc_duplicate, ngram};
use flate2::read::GzDecoder;
use log::info;
use rayon::prelude::*;
use rustc_hash::FxHashSet as HashSet;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

/// Read all files in a directory recursively.
fn read_dir_recursive(dir_path: impl AsRef<Path>) -> Vec<PathBuf> {
    let mut all_paths = Vec::new();
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                all_paths.extend(read_dir_recursive(&path));
            } else {
                all_paths.push(path);
            }
        }
    }
    all_paths.sort_by_key(|path| {
        // split file path with "/"
        let filename = path.file_name().unwrap().to_str().unwrap();
        let numeric_part = filename
            .chars()
            .take_while(|c| c.is_numeric())
            .collect::<String>(); // ファイル名から数字部分を抽出
        numeric_part.parse::<u32>().unwrap_or(0) // 数字部分を数値に変換
    });

    all_paths
}

/// Convert a JSON string to a list of token ids.
fn convert_to_token_ids(line: String) -> Vec<i32> {
    let json_data: Value = serde_json::from_str(&line).expect("Failed to parse JSON");
    if let Some(token_ids) = json_data["token_ids"].as_array() {
        let token_ids: Vec<i32> = token_ids
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|v| v as i32)
            .collect();
        return token_ids;
    }
    Vec::new()
}

/// Search for near-duplicate spans in a document.
fn search(query: &Vec<Vec<i32>>, path: &str, threshold: f32, n: usize) -> Vec<i32> {
    let query_list = query.clone();
    let query_ngram_list = query_list
        .iter()
        .map(|query| ngram(query, n))
        .collect::<Vec<HashSet<usize>>>();

    let file = File::open(path).expect("Failed to open file");
    //let reader = BufReader::new(MultiGzDecoder::new(file));
    let reader = BufReader::new(GzDecoder::new(file));
    let query_num = query_list.len();

    info!("path: {:?} start loading token_ids_list", path);
    let mut token_ids_list = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            let token_ids = convert_to_token_ids(line);
            token_ids_list.push(token_ids);
        }
    }
    info!("loaded token_ids_list");

    // multi thread per query
    let count_list = (0..query_num)
        .into_par_iter()
        .map(|i| {
            let query = &query_list[i];
            let query_ngram = &query_ngram_list[i];
            let mut count = 0;

            for token_ids in &token_ids_list {
                if has_doc_duplicate(token_ids.clone(), &query, &query_ngram, threshold as f64, n) {
                    count += 1;
                }
            }
            info!("query: {:?} count: {:?}", i, count);
            count
        })
        .collect::<Vec<i32>>()
        .try_into()
        .unwrap();
    count_list
}

#[derive(Serialize, Deserialize)]
struct MyData {
    iteration: u32,
    dataset_idx: u32,
    dataset_name: String,
    doc_ids: Vec<u32>,
    text: String,
    token_ids: Vec<u32>,
    completion_stats: CompletionStats,
    metrics: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct CompletionStats {
    count: u32,
    last_iteration: u32,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// directory to search
    #[arg(long, default_value = "path/to/sample_data")]
    search_dir: String,

    /// query path
    #[arg(short, long, default_value = "path/to/sample_data/query.jsonl")]
    query_path: String,

    /// similarity threshold
    #[arg(short, long, default_value_t = 0.6)]
    threshold: f32,

    /// ngram size
    #[arg(short, long, default_value_t = 10)]
    n: usize,

    /// start file idx
    #[arg(long, default_value_t = 0)]
    start_file_idx: usize,

    /// end file idx
    #[arg(long, default_value_t = 142)]
    end_file_idx: usize,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    env::set_var("RUST_LOG", "info");
    env_logger::init();

    // read query
    let file = File::open(&args.query_path)?;
    let reader = BufReader::new(file);
    let mut query_list_all = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        let query: Vec<i32> = serde_json::from_value(v["token_ids"].clone()).unwrap();
        query_list_all.push(query);
    }

    info!("query_list_all: {:?}", query_list_all.len());

    // search path list
    let search_path_list = read_dir_recursive(Path::new(&args.search_dir));
    let search_path_list: Vec<&PathBuf> = search_path_list
        .iter()
        .filter(|path| {
            let parts: Vec<&str> = path.to_str().unwrap().split("/").collect();
            let extracted_part = parts[parts.len() - 1];
            if extracted_part.contains("-") == false {
                return false;
            }
            let file_idx: usize = extracted_part.split("-").collect::<Vec<&str>>()[1]
                .split(".")
                .collect::<Vec<&str>>()[0]
                .parse()
                .unwrap();
            args.start_file_idx * 1000 <= file_idx && file_idx <= args.end_file_idx * 1000
        })
        .collect();
    info!("search_path_list len: {:?}", search_path_list.len());
    let mut count = vec![0; query_list_all.len()];
    for (i, path) in search_path_list.iter().enumerate() {
        let count_per_path = search(
            &query_list_all,
            path.to_str().unwrap(),
            args.threshold,
            args.n,
        );
        for (j, c) in count_per_path.iter().enumerate() {
            count[j] += c;
        }
        info!("path idx: {:?} finished", i);
    }

    info!("count: {:?}", count);

    Ok(())
}

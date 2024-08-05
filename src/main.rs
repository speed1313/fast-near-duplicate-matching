mod lib;

use clap::Parser;
use env_logger;
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

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
    let search_path_list = lib::read_dir_recursive(Path::new(&args.search_dir));
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
        let count_per_path = lib::search(
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

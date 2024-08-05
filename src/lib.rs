use flate2::read::GzDecoder;
use fxhash;
use log::info;
use rayon::prelude::*;
use rustc_hash::FxHashSet as HashSet;
use serde_json::Value;
use std::cmp::max;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

struct RollingHash {
    base: u64,
    modulo: u64,
    hash: u64,
    base_power: u64,
    window_size: usize,
}

impl RollingHash {
    fn new() -> Self {
        Self {
            base: 31,
            modulo: 1_000_000_007,
            hash: 0,
            base_power: 1,
            window_size: 0,
        }
    }

    fn append(&mut self, char: u64) {
        self.hash = (self.hash * self.base + char) % self.modulo;
        self.window_size += 1;
        if self.window_size == 1 {
            self.base_power = 1;
        } else {
            self.base_power = (self.base_power * self.base) % self.modulo;
        }
    }

    fn slide(&mut self, old_char: u64, new_char: u64) {
        self.hash =
            ((self.hash + self.modulo) - (old_char * self.base_power) % self.modulo) % self.modulo;
        self.hash = (self.hash * self.base + new_char) % self.modulo;
    }
}

fn create_frequency_vector<'a>(set: &'a [i32]) -> HashMap<&'a i32, usize> {
    let mut frequency_vector: HashMap<&i32, usize> = HashMap::new();
    for element in set {
        *frequency_vector.entry(element).or_insert(0) += 1;
    }
    frequency_vector
}

fn weighted_jaccard_similarity(x: &HashMap<&i32, usize>, y: &HashMap<&i32, usize>) -> f64 {
    let mut intersection_frequency = 0;
    for (element, &frequency1) in x {
        if let Some(&frequency2) = y.get(element) {
            intersection_frequency += frequency1.min(frequency2);
        }
    }

    let sum_of_frequencies_x: usize = x.values().sum();
    let sum_of_frequencies_y: usize = y.values().sum();
    let union_frequency = sum_of_frequencies_x + sum_of_frequencies_y - intersection_frequency;
    if union_frequency > 0 {
        intersection_frequency as f64 / union_frequency as f64
    } else {
        0.0
    }
}

fn weighted_jaccard(text1: &[i32], text2: &[i32]) -> f64 {
    let x = create_frequency_vector(text1);
    let y = create_frequency_vector(text2);

    let similarity = weighted_jaccard_similarity(&x, &y);

    similarity
}

pub fn ngram(text: &[i32], n: usize) -> HashSet<usize> {
    let mut ngrams = HashSet::default();
    for i in 0..text.len() - n + 1 {
        ngrams.insert(fxhash::hash(&text[i..i + n]));
    }
    ngrams
}

pub fn ngram_rolling(text: &[i32], n: usize) -> HashSet<usize> {
    let mut ngrams = HashSet::default();
    for i in 0..text.len() - n + 1 {
        let text = text.iter().map(|v| *v as u64).collect::<Vec<u64>>();
        let mut rolling_hash = RollingHash::new();
        for c in text[i..i + n].iter() {
            rolling_hash.append(*c);
        }
        ngrams.insert(rolling_hash.hash as usize);
    }
    ngrams
}

// test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_jaccard() {
        let text1 = vec![1, 2, 3, 4, 5];
        let text2 = vec![1, 2, 3, 4, 5];
        assert_eq!(weighted_jaccard(&text1, &text2), 1.0);
        let text1 = vec![1, 2, 2];
        let text2 = vec![1, 1, 2];
        assert_eq!(weighted_jaccard(&text1, &text2), (2.0) / 4.0);
        let text1 = vec![1, 1, 2, 3];
        let text2 = vec![1, 2, 2, 2];
        assert_eq!(weighted_jaccard(&text1, &text2), (2.0) / 6.0);
        let text1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let text2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    }

    #[test]
    fn test_ngram() {
        let text = vec![1, 2, 3, 4, 5];
        let mut ngrams = HashSet::default();
        ngrams.insert(fxhash::hash(&vec![1, 2]));
        ngrams.insert(fxhash::hash(&vec![2, 3]));
        ngrams.insert(fxhash::hash(&vec![3, 4]));
        ngrams.insert(fxhash::hash(&vec![4, 5]));
        assert_eq!(ngram(&text, 2), ngrams);
    }

    #[test]
    fn test_query_contain() {
        let query = vec![1, 2, 3, 4, 5];
        let doc = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let query_ngram = ngram(&query, 10);
        let threshold = 0.8;
        let n = 10;
        assert_eq!(
            has_doc_duplicate(doc, &query, &query_ngram, threshold, n),
            true
        );
    }
    #[test]
    fn test_rolling_hash() {
        let text = vec![1, 2, 3, 4, 5];
        let mut rolling_hash = RollingHash::new();
        for c in text.iter().map(|v| *v as u64) {
            rolling_hash.append(c);
        }
        assert_eq!(
            rolling_hash.hash,
            (1 * u64::pow(31, 4) + 2 * u64::pow(31, 3) + 3 * u64::pow(31, 2) + 4 * 31 + 5)
                % 1_000_000_007
        );
    }

    #[test]
    fn test_update() {
        let text = vec![1, 2, 3, 4, 5];
        let mut rolling_hash = RollingHash::new();
        for c in text.iter().map(|v| *v as u64) {
            rolling_hash.append(c);
        }
        assert_eq!(
            rolling_hash.hash,
            (1 * u64::pow(31, 4) + 2 * u64::pow(31, 3) + 3 * u64::pow(31, 2) + 4 * 31 + 5)
                % 1_000_000_007
        );
        println!("{:?}", rolling_hash.hash);
        rolling_hash.slide(1, 6);
        assert_eq!(
            rolling_hash.hash,
            (2 * u64::pow(31, 4) + 3 * u64::pow(31, 3) + 4 * u64::pow(31, 2) + 5 * 31 + 6)
                % 1_000_000_007
        );
    }
}

pub fn has_doc_duplicate(
    doc: Vec<i32>,
    query: &[i32],
    query_ngram: &HashSet<usize>,
    threshold: f64,
    n: usize,
) -> bool {
    for start in 0..doc.len() - query.len() {
        let is_in_query_ngram = query_ngram.contains(&fxhash::hash(&doc[start..start + n]));
        if !is_in_query_ngram {
            continue;
        }
        let inner_start = max(0, start as i32 - query.len() as i32 + n as i32) as usize;
        for s in inner_start..(start + 1) {
            let end = s + query.len();
            let sim = weighted_jaccard(&query, &doc[s..end]);
            if sim >= threshold {
                return true;
            }
        }
    }
    return false;
}

pub fn has_doc_duplicate_naive(doc: Vec<i32>, query: &[i32], threshold: f64) -> bool {
    for start in 0..doc.len() - query.len() {
        let sim = weighted_jaccard(&query, &doc[start..start + query.len()]);
        if sim >= threshold {
            return true;
        }
    }
    return false;
}

pub fn has_doc_duplicate_rolling(
    doc: Vec<i32>,
    query: &[i32],
    query_ngram: &HashSet<usize>,
    threshold: f64,
    n: usize,
) -> bool {
    let mut rollinghash = RollingHash::new();
    for c in doc[..n].iter().map(|v| *v as u64) {
        rollinghash.append(c);
    }
    for start in 0..doc.len() - query.len() {
        let is_in_query_ngram = query_ngram.contains(&(rollinghash.hash as usize));
        if !is_in_query_ngram {
            // update hash_value
            rollinghash.slide(doc[start] as u64, doc[start + n] as u64);
            continue;
        }
        let inner_start = max(0, start as i32 - query.len() as i32 + n as i32) as usize;
        for s in inner_start..(start + 1) {
            let end = s + query.len();
            let sim = weighted_jaccard(&query, &doc[s..end]);
            if sim >= threshold {
                return true;
            }
        }
        // update hash_value
        rollinghash.slide(doc[start] as u64, doc[start + n] as u64);
    }
    return false;
}

pub fn convert_to_token_ids(line: String) -> Vec<i32> {
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

pub fn read_dir_recursive(dir_path: impl AsRef<Path>) -> Vec<PathBuf> {
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

pub fn search(query: &Vec<Vec<i32>>, path: &str, threshold: f32, n: usize) -> Vec<i32> {
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

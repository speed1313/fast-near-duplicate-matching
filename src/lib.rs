use fxhash;
use rustc_hash::FxHashSet as HashSet;
use std::cmp::max;
use std::collections::HashMap;

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
        let sim_threshold = 0.8;
        let n = 10;
        assert_eq!(
            has_doc_duplicate(doc, &query, &query_ngram, sim_threshold, n),
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
    sim_threshold: f64,
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
            if sim >= sim_threshold {
                return true;
            }
        }
    }
    return false;
}

pub fn has_doc_duplicate_naive(doc: Vec<i32>, query: &[i32], sim_threshold: f64, n: usize) -> bool {
    for start in 0..doc.len() - query.len() {
        let sim = weighted_jaccard(&query, &doc[start..start + query.len()]);
        if sim >= sim_threshold {
            return true;
        }
    }
    return false;
}

pub fn has_doc_duplicate_rolling(
    doc: Vec<i32>,
    query: &[i32],
    query_ngram: &HashSet<usize>,
    sim_threshold: f64,
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
            if sim >= sim_threshold {
                return true;
            }
        }
        // update hash_value
        rollinghash.slide(doc[start] as u64, doc[start + n] as u64);
    }
    return false;
}

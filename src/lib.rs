//! # neardup
//!
//! `neardup` is a library for finding near-duplicate spans in a document.
//! It provides functions to compute n-grams of a text, calculate weighted jaccard similarity, and check whether the document contains spans whose similarity to the query is above a threshold.
//! ## Method
//! ### Fast Near-duplicate Matching
//! - Input: Suffix $s$, document $d$, and $n$ of $n$-gram
//! - Output: Whether $d$ has a span near-duplicate to $s$
//! #### Pseudo-code in Python
//! ```python
//! def fast_near_duplicate_matching(s: list[int], d: list[int], n: int, threshold: float) -> bool:
//!     l_s = len(s)
//!     l_d = len(d)
//!     H = set(ngram(s, n))
//!     for i in range(max(l_d - l_s, 0)):
//!         if d[i:i+n] in H:
//!             for j in range(max(i - l_s + n, 0), i):
//!                 t = d[j:j+l_s]
//!                 if Jaccard_W(s, t) >= threshold:
//!                     return True
//!     return False
//! ```
//!
//! You can use fast hash functions like [fxhash](https://docs.rs/fxhash/latest/fxhash/) or [rolling hash](https://en.wikipedia.org/wiki/Rolling_hash).
//! When the size of $n$ of $n$-gram is small, fxhash is faster than rolling hash. However, when the size of $n$ is large, rolling hash is faster than fxhash because the rolling hash can calculate the hash value of the next $n$-gram in $O(1)$ time.

use fxhash;

use rustc_hash::FxHashSet as HashSet;

use std::cmp::max;
use std::collections::HashMap;

/// A struct for rolling hash.
/// # Examples
/// ```
/// let text = vec![1, 2, 3, 4, 5];
/// let mut rolling_hash = neardup::RollingHash::new();
/// for c in text.iter().map(|v| *v as u64) {
///    rolling_hash.append(c);
/// }
/// assert_eq!(rolling_hash.get_hash(), (1 * u64::pow(31, 4) + 2 * u64::pow(31, 3) + 3 * u64::pow(31, 2) + 4 * 31 + 5) % 1_000_000_007);
/// ```
pub struct RollingHash {
    base: u64,
    modulo: u64,
    hash: u64,
    base_power: u64,
    window_size: usize,
}

impl RollingHash {
    pub fn new() -> Self {
        Self {
            base: 31,
            modulo: 1_000_000_007,
            hash: 0,
            base_power: 1,
            window_size: 0,
        }
    }

    /// Append a character to the window.
    /// # Examples
    /// ```
    /// let text = vec![1, 2, 3, 4, 5];
    /// let mut rolling_hash = neardup::RollingHash::new();
    /// for c in text.iter().map(|v| *v as u64) {
    ///   rolling_hash.append(c);
    /// }
    /// assert_eq!(rolling_hash.get_hash(), (1 * u64::pow(31, 4) + 2 * u64::pow(31, 3) + 3 * u64::pow(31, 2) + 4 * 31 + 5) % 1_000_000_007);
    /// ```
    pub fn append(&mut self, char: u64) {
        self.hash = (self.hash * self.base + char) % self.modulo;
        self.window_size += 1;
        if self.window_size == 1 {
            self.base_power = 1;
        } else {
            self.base_power = (self.base_power * self.base) % self.modulo;
        }
    }

    /// Slide the window by removing the old character and adding the new character.
    /// # Examples
    /// ```
    /// let text = vec![1, 2, 3, 4, 5];
    /// let mut rolling_hash = neardup::RollingHash::new();
    /// for c in text.iter().map(|v| *v as u64) {
    ///     rolling_hash.append(c);
    /// }
    /// rolling_hash.slide(1, 6);
    /// assert_eq!(rolling_hash.get_hash(), (2 * u64::pow(31, 4) + 3 * u64::pow(31, 3) + 4 * u64::pow(31, 2) + 5 * 31 + 6) % 1_000_000_007);
    /// ```
    pub fn slide(&mut self, old_char: u64, new_char: u64) {
        self.hash =
            ((self.hash + self.modulo) - (old_char * self.base_power) % self.modulo) % self.modulo;
        self.hash = (self.hash * self.base + new_char) % self.modulo;
    }

    /// Get the hash value of the current window.
    /// # Examples
    /// ```
    /// let text = vec![1, 2, 3, 4, 5];
    /// let mut rolling_hash = neardup::RollingHash::new();
    /// for c in text.iter().map(|v| *v as u64) {
    ///     rolling_hash.append(c);
    /// }
    /// assert_eq!(rolling_hash.get_hash(), (1 * u64::pow(31, 4) + 2 * u64::pow(31, 3) + 3 * u64::pow(31, 2) + 4 * 31 + 5) % 1_000_000_007);
    /// ```
    pub fn get_hash(&self) -> u64 {
        self.hash
    }
}

fn create_frequency_vector<'a>(set: &'a [i32]) -> HashMap<&'a i32, usize> {
    let mut frequency_vector: HashMap<&i32, usize> = HashMap::new();
    for element in set {
        *frequency_vector.entry(element).or_insert(0) += 1;
    }
    frequency_vector
}

/// Compute weighted jaccard similarity between two texts.
pub fn weighted_jaccard(text1: &[i32], text2: &[i32]) -> f64 {
    let x = create_frequency_vector(text1);
    let y = create_frequency_vector(text2);
    let mut intersection_frequency = 0;
    for (element, frequency1) in &x {
        if let Some(frequency2) = y.get(element) {
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

/// Compute n-grams of a text using fxhash.
///
/// # Examples
///
/// ```
/// let text = vec![1, 2, 3, 4, 5];
/// let ngrams = neardup::ngram(&text, 2);
/// assert_eq!(ngrams.len(), 4);
/// assert_eq!(ngrams.contains(&fxhash::hash(&vec![1, 2])), true);
/// ```
pub fn ngram(text: &[i32], n: usize) -> HashSet<usize> {
    let mut ngrams = HashSet::default();
    for i in 0..text.len() - n + 1 {
        ngrams.insert(fxhash::hash(&text[i..i + n]));
    }
    ngrams
}

/// Compute n-grams of a text using rolling hash.
///
/// # Examples
///
/// ```
/// let text = vec![1, 2, 3, 4, 5];
/// let ngrams = neardup::ngram_rolling(&text, 2);
/// assert_eq!(ngrams.len(), 4);
/// let mut rolling_hash = neardup::RollingHash::new();
/// for c in vec![1, 2].iter().map(|v| *v as u64) {
///    rolling_hash.append(c);
/// }
/// assert_eq!(ngrams.contains(&(rolling_hash.get_hash() as usize)), true);
/// ```
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

        let query_ngram = ngram(&query, 3);
        let threshold = 0.8;
        let n = 3;
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

///  Check whether the document contains spans whose similarity to the query is above a threshold using rabin-karp method with fxhash.
///
/// # Examples
/// ```
/// let query = vec![1, 2, 3, 4, 5];
/// let doc = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let n = 3;
/// let query_ngram = neardup::ngram(&query, n);
/// let sim_threshold = 0.8;
/// assert_eq!(neardup::has_doc_duplicate(doc, &query, &query_ngram, sim_threshold, n), true);
/// ```
///
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

///  Check whether the document contains spans whose similarity to the query is above a threshold using naive method.
///
/// # Examples
/// ```
/// let query = vec![1, 2, 3, 4, 5];
/// let doc = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let sim_threshold = 0.8;
/// assert_eq!(neardup::has_doc_duplicate_naive(doc, &query, sim_threshold), true);
/// ```
pub fn has_doc_duplicate_naive(doc: Vec<i32>, query: &[i32], threshold: f64) -> bool {
    for start in 0..doc.len() - query.len() {
        let sim = weighted_jaccard(&query, &doc[start..start + query.len()]);
        if sim >= threshold {
            return true;
        }
    }
    return false;
}

/// Check whether the document contains spans whose similarity to the query is above a threshold using rabin-karp method with rolling hash.
///
/// # Examples
///
/// ```
/// let query = vec![1, 2, 3, 4, 5];
/// let doc = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let n = 3;
/// let query_ngram = neardup::ngram_rolling(&query, n);
/// let sim_threshold = 0.8;
/// assert_eq!(neardup::has_doc_duplicate_rolling(doc, &query, &query_ngram, sim_threshold, n), true);
/// ```
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

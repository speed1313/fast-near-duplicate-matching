# Fast Near-duplicate Matching  &emsp; [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/neardup.svg
[crates.io]: https://crates.io/crates/neardup



Fast near-duplicate matching is a method for quickly finding near-duplicate spans in a document by utilizing the [Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm). This algorithm can be used to count the near-duplicates of a query in a pre-training corpus, facilitating the analysis of memorization in Large Language Models (LLMs).

This repository contains the implementation of the fast near-duplicate matching algorithm and the benchmark for the algorithm in Rust. The core functionalities is provided as a crate [```neardup```]().

## Method
### Fast Near-duplicate Matching
- Input: Suffix $s$, document $d$, and $n$ of $n$-gram
- Output: Whether $d$ has a span near-duplicate to $s$

#### Pseudo-code in Python
```python
def fast_near_duplicate_matching(s: list[int], d: list[int], n: int, threshold: float) -> bool:
    l_s = len(s)
    l_d = len(d)
    H = set(ngram(s, n))
    for i in range(max(l_d - l_s, 0)):
        if d[i:i+n] in H:
            for j in range(max(i - l_s + n, 0), i):
                t = d[j:j+l_s]
                if Jaccard_W(s, t) >= threshold:
                    return True
    return False
```

You can use fast hash functions like [fxhash](https://docs.rs/fxhash/latest/fxhash/) or [rolling hash](https://en.wikipedia.org/wiki/Rolling_hash).

When the size of $n$ of $n$-gram is small, fxhash is faster than rolling hash. However, when the size of $n$ is large, rolling hash is faster than fxhash because the rolling hash can calculate the hash value of the next $n$-gram in $O(1)$ time.



## How to run
You can count the near-duplicates of queries in the document (sample queries (2 queries) and documents (100 documents) are in the `sample_data` folder) by running the following command:

```bash
$ cargo run --release -- --search-dir ./sample_data --query-path ./sample_data/query.jsonl --threshold 0.6 --n 10
[2024-08-07T10:59:40Z INFO  neardup] query_list_all: 2
[2024-08-07T10:59:40Z INFO  neardup] search_path_list len: 1
[2024-08-07T10:59:40Z INFO  neardup] path: "./sample_data/pythia-00000-00999.jsonl.gz" start loading token_ids_list
[2024-08-07T10:59:40Z INFO  neardup] loaded token_ids_list
[2024-08-07T10:59:40Z INFO  neardup] query: 0 count: 1
[2024-08-07T10:59:40Z INFO  neardup] query: 1 count: 1
[2024-08-07T10:59:40Z INFO  neardup] path idx: 0 finished
[2024-08-07T10:59:40Z INFO  neardup] count: [1, 1]
```

### Count near-duplicates in the Pythia dataset
You can download the Pythia dataset from [here](https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset)
After downloading the dataset, you can convert the dataset to the format that this program can read by running the following command:
```bash
$ python scripts/prepare_pythia.py --output_dir path/to/output/folder --pythia_data_path path/to/merged/folder/document
```
Then, you can count the near-duplicates of queries in the Pythia dataset by running the following command:
```bash
$ cargo run --release -- --search-dir path/to/output/folder --query-path path/to/output/folder/query.jsonl --threshold 0.6 --n 10
```


## Benchmark
You can run the benchmark for the three methods:
- Fast near-duplicate matching with **fxhash**
- Fast near-duplicate matching with **rolling hash**
- **Naive** near-duplicate matching
```bash
$ cargo bench
```



## References
- [Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm)
- [fxhash](https://docs.rs/fxhash/latest/fxhash/)
- [Pythia](https://github.com/EleutherAI/pythia)


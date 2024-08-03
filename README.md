# Fast Near-duplicate Matching

Fast near-duplicate matching is a method for quickly finding near-duplicate spans in a document by utilizing the [Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm). This algorithm can be used to count the near-duplicates of a query in a pre-training corpus, facilitating the analysis of memorization in Large Language Models (LLMs).

This repository contains the implementation of the fast near-duplicate matching algorithm and the benchmark for the algorithm in Rust.


## Method
### Fast Near-duplicate Matching
- Input: Suffix $s$, document $d$, and $n$ of $n$-gram
- Output: Whether $d$ has a span near-duplicate to $s$

#### Pseudo-code in Python
```python
def fast_near_duplicate_matching(s: list[int], d: list[int], n: int, delta: float) -> bool:
    l_s = len(s)
    l_d = len(d)
    H = set(ngram(s, n))
    for i in range(max(l_d - l_s, 0)):
        if d[i:i+n] in H:
            for j in range(max(i - l_s + n, 0), i):
                t = d[j:j+l_s]
                if Jaccard_W(s, t) >= delta:
                    return True
    return False
```

You can use fast hash functions like [fxhash](https://docs.rs/fxhash/latest/fxhash/) or [rolling hash](https://en.wikipedia.org/wiki/Rolling_hash).

When the size of $n$ of $n$-gram is small, fxhash is faster than rolling hash. However, when the size of $n$ is large, rolling hash is faster than fxhash because the rolling hash can calculate the hash value of the next $n$-gram in $O(1)$ time.



## How to run
You can calculate the near-duplicates of a query in a random document using three methods:
- Fast near-duplicate matching with fxhash
- Fast near-duplicate matching with rolling hash
- Naive near-duplicate matching

```bash
cargo run --release -- --sim-threshold 0.6 --ngram-size 10
```

## Benchmark
You can run the benchmark for the three methods using the following command:
```bash
cargo bench
```



## References
- [Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm)
- [fxhash](https://docs.rs/fxhash/latest/fxhash/)


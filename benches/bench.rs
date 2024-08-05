use fast_near_duplicate_matching as lib;

use rand::Rng;

use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let threshold = 0.6;
    let n = 10;
    let query_len = 50;
    let mut rng = rand::thread_rng();
    let query_num = 30000;
    let mut queries = Vec::new();
    for _ in 0..query_num {
        let query = (0..query_len)
            .map(|_| rng.gen_range(0..50254))
            .collect::<Vec<i32>>();
        queries.push(query);
    }

    let doc = (0..2048)
        .map(|_| rng.gen_range(0..50254))
        .collect::<Vec<i32>>();

    let sub_doc = &doc[24..24 + query_len];
    queries.push(sub_doc.to_vec());
    for i in 0..5 {
        let mut copy_sub_doc = sub_doc.to_vec();
        for j in 0..5 {
            let random_idx = rng.gen_range(0..query_len);
            copy_sub_doc[random_idx] = 0;
        }
        queries.push(copy_sub_doc);
    }

    c.bench_function("has_doc_duplicate", |b| {
        b.iter(|| {
            let ngram = lib::ngram(&queries[0], n);
            lib::has_doc_duplicate(doc.clone(), &queries[0], &ngram, threshold as f64, n)
        })
    });
    c.bench_function("has_doc_duplicate_rolling", |b| {
        b.iter(|| {
            let ngram = lib::ngram_rolling(&queries[0], n);
            lib::has_doc_duplicate_rolling(doc.clone(), &queries[0], &ngram, threshold as f64, n)
        })
    });
    // c.bench_function("has_doc_duplicate_naive", |b| {
    //     b.iter(|| {
    //         lib::has_doc_duplicate_naive(
    //             doc.clone(),
    //             &queries[0],
    //             args.threshold as f64,
    //             args.n,
    //         )
    //     })
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

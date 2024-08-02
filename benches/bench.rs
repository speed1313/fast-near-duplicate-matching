use fast_near_duplicate_matching as lib;

use clap::Parser;
use rand::Rng;

use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// similarity threshold
    #[arg(short, long, default_value_t = 0.8)]
    sim_threshold: f32,

    /// ngram size
    #[arg(short, long, default_value_t = 10)]
    ngram_size: usize,
}

fn criterion_benchmark(c: &mut Criterion) {
    let args = Args::parse();
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
    let mut sub_doc = sub_doc.to_vec();
    for _ in 0..5 {
        let idx = rng.gen_range(0..query_len);
        sub_doc[idx] = 0;
    }
    queries.push(sub_doc);

    c.bench_function("has_doc_duplicate", |b| {
        b.iter(|| {
            let ngram = lib::ngram(&queries[0], args.ngram_size);
            lib::has_doc_duplicate(
                doc.clone(),
                &queries[0],
                &ngram,
                args.sim_threshold as f64,
                args.ngram_size,
            )
        })
    });
    c.bench_function("has_doc_duplicate_rolling", |b| {
        b.iter(|| {
            let ngram = lib::ngram_rolling(&queries[0], args.ngram_size);
            lib::has_doc_duplicate_rolling(
                doc.clone(),
                &queries[0],
                &ngram,
                args.sim_threshold as f64,
                args.ngram_size,
            )
        })
    });
    // c.bench_function("has_doc_duplicate_naive", |b| {
    //     b.iter(|| {
    //         lib::has_doc_duplicate_naive(
    //             doc.clone(),
    //             &queries[0],
    //             args.sim_threshold as f64,
    //             args.ngram_size,
    //         )
    //     })
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

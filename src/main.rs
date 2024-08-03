mod lib;

use clap::Parser;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// similarity threshold
    #[arg(short, long, default_value_t = 0.6)]
    sim_threshold: f32,

    /// ngram size
    #[arg(short, long, default_value_t = 10)]
    ngram_size: usize,
}

fn main() -> std::io::Result<()> {
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

    let mut matched_query = Vec::new();

    let now = std::time::Instant::now();
    let mut count = 0;
    for query in &queries {
        let ngram = lib::ngram(&query, args.ngram_size);
        let is_matching = lib::has_doc_duplicate(
            doc.clone(),
            &query,
            &ngram,
            args.sim_threshold as f64,
            args.ngram_size,
        );
        if is_matching {
            count += 1;
            matched_query.push(query);
        }
    }
    println!("elapsed: {:?}", now.elapsed());
    println!("count: {}", count);
    println!("matched_query: {:?}", matched_query);

    // rolling hash test
    let mut count = 0;
    let mut matched_query = Vec::new();
    let now = std::time::Instant::now();
    for query in &queries {
        let ngram = lib::ngram_rolling(&query, args.ngram_size);
        let is_matching = lib::has_doc_duplicate_rolling(
            doc.clone(),
            &query,
            &ngram,
            args.sim_threshold as f64,
            args.ngram_size,
        );
        if is_matching {
            count += 1;
            matched_query.push(query);
        }
    }
    println!("elapsed: {:?}", now.elapsed());
    println!("count: {}", count);
    println!("matched_query: {:?}", matched_query);

    // rolling hash test
    let mut count = 0;
    let mut matched_query = Vec::new();
    let now = std::time::Instant::now();
    for query in &queries {
        let is_matching =
            lib::has_doc_duplicate_naive(doc.clone(), &query, args.sim_threshold as f64);
        if is_matching {
            count += 1;
            matched_query.push(query);
        }
    }
    println!("elapsed: {:?}", now.elapsed());
    println!("count: {}", count);
    println!("matched_query: {:?}", matched_query);

    Ok(())
}

use std::time::Instant;

use crate::matrix::{Matrix, matmul};

pub fn benchmark_matmul() {
    let mut times = Vec::new();
    for _ in 0..3 {
        let a = Matrix::random(1000, 500, -1.0, 1.0);
        let b = Matrix::random(500, 1000, -1.0, 1.0);
        let start = Instant::now();
        let c = matmul(&a, &b);
        let end = Instant::now();
        times.push(end.duration_since(start).as_nanos());
    }

    println!(
        "Time taken: {:?} s",
        (times.iter().sum::<u128>() / times.len() as u128) as f64 / 1_000_000_000.0
    );
}

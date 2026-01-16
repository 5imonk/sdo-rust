use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
// Import the internal modules directly for benchmarks
use sdo::obs::Observer;
use sdo::obset::ObserverSet;
use std::time::Duration;

fn create_test_observer(index: usize, data: Vec<f64>, observations: f64) -> Observer {
    Observer {
        data,
        observations,
        time: index as f64,
        age: 1.0,
        index,
        label: None,
        cluster_observations: Vec::new(),
    }
}

fn create_test_observer_set(size: usize, dimensions: usize) -> ObserverSet {
    let mut obset = ObserverSet::new();

    for i in 0..size {
        let data: Vec<f64> = (0..dimensions)
            .map(|j| (i * dimensions + j) as f64)
            .collect();
        let observations = (size - i) as f64;
        let observer = create_test_observer(i, data, observations);
        obset.insert(observer);
    }

    obset.set_num_active(size / 2);
    obset
}

fn benchmark_distance_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_insertion");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000, 2000].iter() {
        let mut obset = create_test_observer_set(*size, 5);
        let new_data = vec![999.0, 1000.0, 1001.0, 1002.0, 1003.0];
        let new_observer = create_test_observer(*size, new_data.clone(), 50.0);

        group.bench_with_input(BenchmarkId::new("optimized_insert", size), size, |b, _| {
            b.iter(|| {
                let mut test_obset = obset.clone();
                let test_observer = create_test_observer(*size, new_data.clone(), 50.0);
                test_obset.insert(test_observer);
                black_box(test_obset.len())
            });
        });
    }

    group.finish();
}

fn benchmark_neighbor_finding(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_finding");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000, 2000].iter() {
        let obset = create_test_observer_set(*size, 5);

        for k in [5, 10, 50].iter() {
            group.bench_with_input(
                BenchmarkId::new("k_nearest_neighbors", format!("size_{}_k_{}", size, k)),
                &(size, k),
                |b, (_, _)| {
                    b.iter(|| {
                        let neighbors = obset.get_k_nearest_neighbors(0, *k);
                        black_box(neighbors.len())
                    });
                },
            );
        }

        for threshold in [1.0, 10.0, 100.0].iter() {
            group.bench_with_input(
                BenchmarkId::new(
                    "neighbors_within_threshold",
                    format!("size_{}_thresh_{}", size, threshold),
                ),
                &(size, threshold),
                |b, (_, _)| {
                    b.iter(|| {
                        let neighbors = obset.get_neighbors_within_threshold(0, *threshold);
                        black_box(neighbors.len())
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_threshold_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_computation");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000, 2000].iter() {
        let mut obset = create_test_observer_set(*size, 5);

        // Benchmark uncached computation
        group.bench_with_input(
            BenchmarkId::new("uncached_threshold", size),
            size,
            |b, _| {
                b.iter(|| {
                    for i in 0..10.min(*size) {
                        let threshold = obset.compute_local_threshold_cached(i, 5);
                        black_box(threshold);
                    }
                });
            },
        );

        // Benchmark cached computation
        group.bench_with_input(BenchmarkId::new("cached_threshold", size), size, |b, _| {
            b.iter(|| {
                obset.validate_threshold_cache();
                for i in 0..10.min(*size) {
                    let threshold = obset.compute_local_threshold_cached(i, 5);
                    black_box(threshold);
                }
            });
        });
    }

    group.finish();
}

fn benchmark_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering");
    group.measurement_time(Duration::from_secs(30));

    for size in [50, 100, 200, 500].iter() {
        let mut obset = create_test_observer_set(*size, 5);

        group.bench_with_input(BenchmarkId::new("learn_cluster", size), size, |b, _| {
            b.iter(|| {
                let mut test_obset = obset.clone();
                let clusters = test_obset.learn_cluster(5, 0.5, 3, false);
                black_box(clusters.len())
            });
        });
    }

    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000].iter() {
        let mut obset = create_test_observer_set(*size, 5);
        let batch_size = size / 10;
        let updated_indices: Vec<usize> = (0..batch_size).collect();

        group.bench_with_input(BenchmarkId::new("batch_update", size), size, |b, _| {
            b.iter(|| {
                let mut test_obset = obset.clone();
                test_obset.batch_update_distance_lists(&updated_indices);
                black_box(test_obset.len())
            });
        });
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(5));

    // Test memory efficiency with increasing sizes
    for size in [100, 500, 1000, 2000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_observer_set", size),
            size,
            |b, size| {
                b.iter(|| {
                    let obset = create_test_observer_set(*size, 5);
                    black_box(obset.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_distance_insertion,
    benchmark_neighbor_finding,
    benchmark_threshold_computation,
    benchmark_clustering,
    benchmark_batch_operations,
    benchmark_memory_usage
);

criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
// Import the internal modules directly for benchmarks
use sdo::obs::Observer;
use sdo::obset::ObserverSet;
use sdo::sdostrcl_impl::SDOstreamclust;
use sdo::utils::{sample_random_matrix_uniform_unit, DistanceMetric};
use std::collections::HashMap;
use std::time::Duration;

fn create_test_observer(index: usize, data: Vec<f64>, observations: f64) -> Observer {
    Observer {
        data,
        observations,
        time: index as f64,
        age: 1.0,
        index,
        local_threshold: 0.0,
        label_observations: HashMap::new(),
        label_time: 0.0,
    }
}

fn create_test_observer_set(
    size: usize,
    dimensions: usize,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
) -> ObserverSet {
    let mut obset = ObserverSet::new(distance_metric, minkowski_p);

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
        let obset = create_test_observer_set(*size, 5, DistanceMetric::Euclidean, None);
        let new_data = vec![999.0, 1000.0, 1001.0, 1002.0, 1003.0];

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
        let obset = create_test_observer_set(*size, 5, DistanceMetric::Euclidean, None);

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

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000].iter() {
        let obset = create_test_observer_set(*size, 5, DistanceMetric::Euclidean, None);
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
                    let obset = create_test_observer_set(*size, 5, DistanceMetric::Euclidean, None);
                    black_box(obset.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SDOstreamclust learn_impl for profiling (e.g. cargo flamegraph).
fn benchmark_sdostreamclust_learn_impl(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdostreamclust_learn_impl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let k = 200;
    let t_fading = 100.0;
    let t_sampling = 100.0;
    let x = 3;
    let rho = 0.2;
    let chi_min = 1;
    let chi_prop = 0.05;
    let zeta = 0.7;
    let min_cluster_size = 5;
    let dimension = 2;

    for block_size in [25, 50, 100].iter() {
        let points = sample_random_matrix_uniform_unit(dimension, *block_size);
        let times: Vec<f64> = (0..*block_size).map(|i| i as f64).collect();

        group.bench_with_input(
            BenchmarkId::new("learn_impl", block_size),
            block_size,
            |b, _| {
                b.iter(|| {
                    let mut model = SDOstreamclust::new_for_benchmark(
                        k,
                        t_fading,
                        t_sampling,
                        x,
                        rho,
                        chi_min,
                        chi_prop,
                        zeta,
                        min_cluster_size,
                        dimension,
                    );
                    model.learn_impl(&points, &times);
                    black_box(())
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
    benchmark_batch_operations,
    benchmark_memory_usage,
    benchmark_sdostreamclust_learn_impl
);

criterion_main!(benches);

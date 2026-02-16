//! Profiling Benchmarks für Performance-Analyse mit cargo flamegraph oder perf.
//! Fokussiert auf die wichtigsten Hot Paths: KNN-Suche, Distanzberechnung, Batch-Operationen.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sdo::obs::Observer;
use sdo::obset::ObserverSet;
use sdo::sdostream_impl::SDOstream;
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
) -> ObserverSet {
    let mut obset = ObserverSet::new(distance_metric, None);
    for i in 0..size {
        let data: Vec<f64> = (0..dimensions)
            .map(|j| ((i * dimensions + j) as f64) * 0.1)
            .collect();
        let observations = (size - i) as f64;
        let observer = create_test_observer(i, data, observations);
        obset.insert(observer);
    }
    obset.set_num_active(size / 2);
    obset
}

/// Profiling: search_neighbors_unified (single point)
fn benchmark_search_neighbors_unified(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_neighbors_unified");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    for size in [200, 500, 1000].iter() {
        let obset = create_test_observer_set(*size, 128, DistanceMetric::Euclidean);
        let query_point: Vec<f64> = (0..128).map(|i| (i as f64) * 0.1).collect();

        group.bench_with_input(
            BenchmarkId::new("single_point", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = obset.search_neighbors_unified(&query_point, 10, None, None);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

/// Profiling: search_neighbors_unified_batch (batch points)
fn benchmark_search_neighbors_unified_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_neighbors_unified_batch");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for (size, batch_size) in [(200, 25), (500, 50), (1000, 100)].iter() {
        let obset = create_test_observer_set(*size, 128, DistanceMetric::Euclidean);
        let batch_points: Vec<Vec<f64>> = (0..*batch_size)
            .map(|i| {
                (0..128)
                    .map(|j| ((i * 128 + j) as f64) * 0.1)
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch", format!("obs_{}_batch_{}", size, batch_size)),
            &(size, batch_size),
            |b, _| {
                b.iter(|| {
                    let result = obset.search_neighbors_unified_batch(&batch_points, 10, None, None);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

/// Profiling: SDOstream learn_impl (full pipeline)
fn benchmark_sdostream_learn_impl(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdostream_learn_impl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let k = 200;
    let dimension = 128;
    for block_size in [25, 50, 100].iter() {
        let points = sample_random_matrix_uniform_unit(dimension, *block_size);
        let times: Vec<f64> = (0..*block_size).map(|i| i as f64).collect();

        group.bench_with_input(
            BenchmarkId::new("learn_impl", block_size),
            block_size,
            |b, _| {
                b.iter(|| {
                    let mut model = SDOstream::new_for_benchmark(
                        k,
                        100.0,
                        100.0,
                        3,
                        0.2,
                        dimension,
                    );
                    model.learn_impl_public(&points, &times);
                    black_box(())
                });
            },
        );
    }
    group.finish();
}

/// Profiling: DistanceMatrix rebuild
fn benchmark_distance_matrix_rebuild(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix_rebuild");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for size in [100, 200, 500].iter() {
        let mut obset = create_test_observer_set(*size, 64, DistanceMetric::Euclidean);

        group.bench_with_input(
            BenchmarkId::new("rebuild", size),
            size,
            |b, _| {
                b.iter(|| {
                    obset.rebuild_distance_lists_public();
                    black_box(())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    profiling_benches,
    benchmark_search_neighbors_unified,
    benchmark_search_neighbors_unified_batch,
    benchmark_sdostream_learn_impl,
    benchmark_distance_matrix_rebuild
);

criterion_main!(profiling_benches);

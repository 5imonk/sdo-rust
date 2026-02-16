# Profiling Troubleshooting Guide

## Problem: perf_event_paranoid Error

Wenn du diese Fehlermeldung siehst:
```
Access to performance monitoring and observability operations is limited.
perf_event_paranoid setting is 4
```

### Lösung 1: Temporär (empfohlen für Tests)

```bash
# Als root ausführen
sudo sysctl -w kernel.perf_event_paranoid=1
```

Dann erneut versuchen:
```bash
./scripts/profile.sh flamegraph benchmark_sdostream_learn_impl
```

### Lösung 2: Permanente Lösung

```bash
# Als root
echo "kernel.perf_event_paranoid = 1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Lösung 3: Als Root ausführen

```bash
sudo ./scripts/profile.sh flamegraph benchmark_sdostream_learn_impl
```

### Lösung 4: Alternative ohne perf (langsamer)

Wenn perf nicht verfügbar ist, kannst du stattdessen Benchmarks verwenden:

```bash
# Normale Benchmarks (ohne Flamegraph)
./scripts/profile.sh bench benchmark_sdostream_learn_impl

# Oder einfache Ausgabe
./scripts/profile.sh bench-simple benchmark_search_neighbors_unified_batch
```

Die Ergebnisse werden in `target/criterion/` gespeichert mit HTML-Reports.

## Alternative Profiling-Methoden

### 1. Criterion Benchmarks (funktioniert immer)

```bash
cargo bench --bench profiling_benchmarks
```

Ergebnisse in `target/criterion/` mit interaktiven HTML-Reports.

### 2. Python cProfile (für Python-Code)

```python
import cProfile
import pstats
from sdo import SDOstream
import numpy as np

model = SDOstream(k=200, x=3, dimension=128)
data = np.random.rand(1000, 128).astype(np.float64)
times = np.arange(1000, dtype=np.float64)

profiler = cProfile.Profile()
profiler.enable()

for i in range(0, 1000, 50):
    model.learn(data[i:i+50], time=times[i:i+50])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### 3. Valgrind (Linux, sehr detailliert)

```bash
valgrind --tool=callgrind \
  cargo run --release --bench profiling_benchmarks -- benchmark_sdostream_learn_impl

# Analyse mit KCacheGrind
kcachegrind callgrind.out.*
```

## Überprüfen der Systemeinstellungen

```bash
# Aktueller Wert
cat /proc/sys/kernel/perf_event_paranoid

# Verfügbare Werte:
# -1: Alle Events erlaubt (unsicher)
#  0: Raw/ftrace Events erlaubt
#  1: CPU Events erlaubt (empfohlen für Profiling)
#  2: Nur User-Space Events
#  3+: Sehr restriktiv
```

## Weitere Hilfe

- [Flamegraph Documentation](https://github.com/flamegraph-rs/flamegraph)
- [Perf Security Guide](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)

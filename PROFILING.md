# Profiling (Rust)

Kurze Anleitung für Sampling-Profiling und Flamegraphs der Rust-Erweiterung.

## 1. Build mit Debuginfo

Damit in Flamegraphs Rust-Funktionsnamen erscheinen, die Extension mit dem Profil `prof` bauen (Release-Optimierungen + Debuginfo):

```bash
# Nur Library (z.B. für Benchmarks)
cargo build --profile prof

# Python-Extension (maturin nutzt standardmäßig release; für prof siehe pyproject.toml / maturin docs)
maturin develop --release
# Optional: In Cargo.toml existiert [profile.prof]; für maturin mit prof ggf. Umgebungsvariable oder
# manuell: cargo build --profile prof, dann .so ins venv kopieren
```

Das Cargo-Profil `prof` ist in `Cargo.toml` definiert:

```toml
[profile.prof]
inherits = "release"
debug = true
```

## 2. Flamegraph beim Aufruf von Python

**Linux (perf):**

1. Extension mit Debuginfo bauen (s.o.).
2. Python-Skript unter perf ausführen:
   ```bash
   perf record -g python python/sdostreamclust/visualize_sdostreamclust.py
   ```
   (oder ein kürzeres Skript, das nur viele `learn()`-Aufrufe macht.)
3. Flamegraph erzeugen (z.B. mit [inferno](https://github.com/jonhoo/inferno)):
   ```bash
   perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
   ```
   Oder mit `cargo install flamegraph`:
   ```bash
   perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
   ```

So siehst du, welche Rust-Funktionen (z.B. `learn_impl`, `predict_point`, `sample_impl`, `fit_impl`, `update`) wie viel Anteil an der Laufzeit haben.

## 3. Flamegraph nur Rust (Benchmark)

Ohne Python-Runtime: Benchmark ausführen und mit Flamegraph profilen.

**Voraussetzung:** `cargo install flamegraph` (und unter Linux ggf. `perf`-Rechte, z.B. `kernel.perf_event_paranoid`).

```bash
# Benchmark profilen (Standard: release; für Debug-Symbole zuerst mit prof bauen)
cargo flamegraph --bench optimization_benchmarks
```

Nur den SDOstreamclust-Benchmark ausführen (Criterion-Filter):

```bash
cargo flamegraph --bench optimization_benchmarks -- sdostreamclust_learn_impl
```

Mit Profil `prof` (Debuginfo für lesbare Funktionsnamen in der SVG):

```bash
cargo build --bench optimization_benchmarks --profile prof
# Anschließend das Binary unter perf ausführen oder flamegraph auf target/prof/... anwenden
perf record -g target/prof/optimization_benchmarks sdostreamclust_learn_impl
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

Der Benchmark `sdostreamclust_learn_impl` ruft SDOstreamclust mit Testdaten und mehrfach `learn_impl` auf; die Ergebnisse eignen sich zum Vergleichen von Blockgrößen (25, 50, 100).

## 4. Nützliche Tools

- **perf** (Linux): `perf record -g`, `perf report`
- **inferno**: `cargo install flamegraph` liefert u.a. `inferno-collapse-perf`, `inferno-flamegraph`
- **cargo flamegraph**: `cargo install flamegraph`; vereinfacht Aufnahme + Erzeugung des Flamegraphs

## 5. Troubleshooting (Linux)

### „Access to performance monitoring … is limited“ / perf_event_paranoid

Wenn `cargo flamegraph` oder `perf record` mit einer Meldung zu **perf_event_paranoid** abbricht, blockiert der Kernel Nutzer-Profiling (typisch bei `perf_event_paranoid = 4`).

**Temporär lockern (erfordert root):**
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```
oder
```bash
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**Dauerhaft:** In `/etc/sysctl.conf` oder z. B. `/etc/sysctl.d/99-perf.conf` eintragen:
```
kernel.perf_event_paranoid = -1
```
Danach `sudo sysctl -p` bzw. Reboot.

**Alternative ohne Kernel-Änderung:** Benchmark-Binary mit root unter perf ausführen:
```bash
cargo build --bench optimization_benchmarks --profile bench
sudo perf record -g target/bench/optimization_benchmarks sdostreamclust_learn_impl
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```
(Recht nur für lokales Profiling nutzen.)

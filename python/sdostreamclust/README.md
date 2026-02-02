# SDOstreamclust – Tests und Visualisierung

Von **Projektroot** aus:

- **Tests (SDOstreamclust):**  
  `python python/sdostreamclust/test_sdostreamclust.py`
- **Tests (SDOstream – Streaming Outlier Detection):**  
  `python python/sdostream/test_sdostream.py`
- **Visualisierung (Streaming + Frames/Video):**  
  `python python/sdostreamclust/visualize_sdostreamclust.py`  
  Ausgaben: `python/sdostreamclust/out/` (Frames, Video, CSV).
- **Streaming-Evaluation (ARI, No-Warmup):**  
  `python python/sdostreamclust/run_streaming_clustering_eval.py [--data-folders ...] [--output python/sdostreamclust/out]`  
  Ergebnisse: `python/sdostreamclust/out/streaming_results.csv` (Default).

Voraussetzung: `maturin develop` im Projektroot. Optional: `scikit-learn`, `matplotlib`, `scipy`, `moviepy` (für Video), `pandas`.

import sys
import os

# Add paths for sdo module
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append('/home/simon/sdo/.venv/lib/python3.12/site-packages')

# Add paths for sdo module and ensure we use virtual environment
sys.path.insert(0, '/home/simon/sdo/.venv/lib/python3.12/site-packages')

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Lade Daten (verwende zuerst concept_drift.arff)
data_loaded = False
data_paths = [
    'evaluation_tests/data/example/concept_drift.arff',  # Default dataset for concept drift
    # Fallback-Optionen für andere Dateien
    'evaluation_tests/data/example/*.arff',
    'example/dataset.csv',
]

for data_path in data_paths:
    try:
        if data_path.endswith('.arff'):
            # Lade ARFF-Datei
            import glob
            arff_files = glob.glob(data_path)
            if arff_files:
                filename = arff_files[0]  # Nimm erste gefundene Datei
                print(f"Lade ARFF-Datei: {filename}")
                arffdata = loadarff(filename)
                df_data = pd.DataFrame(arffdata[0])
                
                # Konvertiere class-Spalte falls nötig
                if df_data['class'].dtypes == 'object':
                    df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
                
                y = df_data['class'].to_numpy()
                t = np.arange(len(y), dtype=np.float64)
                df_data.drop(columns=['class'], inplace=True)
                x = df_data.to_numpy().astype(np.float64)
                
                # Normalisiere Daten
                scaler = MinMaxScaler()
                x = scaler.fit_transform(x)
                
                # Konvertiere Labels zu int (falls String/Bytes)
                if y.dtype == object:
                    # Konvertiere Bytes zu String falls nötig
                    if isinstance(y[0], bytes):
                        y = np.array([val.decode('utf-8') for val in y])
                    # Konvertiere String-Labels zu int (-1 für Outlier bleibt -1)
                    y_int = []
                    for val in y:
                        try:
                            y_int.append(int(val))
                        except ValueError:
                            # Falls Label nicht konvertierbar, verwende LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            y_int = y
                            break
                    if len(y_int) == len(y):
                        y = np.array(y_int, dtype=int)
                else:
                    y = y.astype(int)
                
                data_loaded = True
                print(f"✓ Daten geladen: {len(x)} Punkte, {x.shape[1]} Dimensionen")
                break
        else:
            # Lade CSV-Datei
            df = pd.read_csv(data_path)
            t = df['timestamp'].to_numpy()
            x = df[['f0','f1']].to_numpy()
            y = df['label'].to_numpy()
            # Normalisiere Daten
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            data_loaded = True
            print(f"✓ Daten geladen: {len(x)} Punkte, {x.shape[1]} Dimensionen")
            break
    except (FileNotFoundError, KeyError, ValueError) as e:
        continue

if not data_loaded:
    print("Warnung: Keine Daten gefunden. Generiere Beispieldaten...")
    np.random.seed(42)
    n_points = 1000
    t = np.arange(n_points, dtype=np.float64)
    
    # Generiere zwei Cluster und einige Outlier
    cluster1 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([8.0, 8.0])
    x = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # Labels: 0 für Cluster 1, 1 für Cluster 2
    y = np.concatenate([np.zeros(n_points // 2), np.ones(n_points // 2)]).astype(int)

k = 400 # Model size (larger for streaming evaluation)
T = 400 # Time Horizon (larger t_fading for better adaptation)
T_sampling = 250 # Sampling Interval (more frequent updates)
x_neighbors = 5 # Anzahl nächster Nachbarn (slightly increased for streaming)
chi_min = 1 # Minimum chi value
chi_prop = 0.01 # Proportional chi (10% of k, slightly increased for streaming)
zeta = 0.6 # Mixing-Parameter
min_cluster_size = 2 # Minimale Clustergröße

# Initialize SDOstreamclust for streaming evaluation (no warmup)
print("About to import SDOstreamclust...")
try:
    from sdo import SDOstreamclust
    print("SDOstreamclust module imported successfully")
    classifier = SDOstreamclust(
        k=k, 
        x=x_neighbors, 
        t_fading=T,
        t_sampling=T,
        chi_min=chi_min,
        chi_prop=chi_prop,
        zeta=zeta,
        min_cluster_size=min_cluster_size,
        dimension=x.shape[1]  # Provide dimension for no warmup
    )
except ImportError as e:
    print(f"Error: Could not import sdo module: {e}")
    print("Please install the module with 'maturin develop' or 'pip install .'")
    sys.exit(1)

all_predic = []
all_scores = []

block_size = 1 # per-point processing

# Streaming-Verarbeitung (no warmup - direct learning)
for i in range(0, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size] if len(t) > 0 else None
    
    # Direkter Punkt-Streaming ohne Vorhersage
    try:
        label, score = classifier.learn(chunk[0:1])  # Process first point in chunk
        all_predic.append(label)
        all_scores.append(score)
    except Exception as e:
        print(f"Error processing point {i}: {e}")
        all_predic.append(-1)
        all_scores.append(0.0)
    
    # Fortschritt nach jedem Punkt
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{x.shape[0]} points...")

p = np.array(all_predic) # clustering labels
s = np.array(all_scores) # outlierness scores
s = -1/(s+1) # norm. to avoid inf scores

# Print final summary for streaming evaluation
try:
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import roc_auc_score
    print(f"\n=== Final Results ===")
    print(f"Processed {len(all_predic)} points with no warmup phase")
    print(f"Final Adjusted Rand Index: {adjusted_rand_score(y[:len(all_predic)], all_predic):.4f}")
    print(f"Final ROC AUC score: {roc_auc_score(y[:len(all_predic)]<0, all_scores):.4f}")

    # Thresholding top outliers based on Chebyshev's inequality (88.9%)
    th = np.mean(all_scores)+3*np.std(all_scores)
    all_scores_np = np.array(all_scores)
    p_final = all_scores_np.copy()
    p_final[all_scores_np>th] = -1
except ImportError as e:
    print(f"Error importing evaluation metrics: {e}")

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score

print("Adjusted Rand Index (clustering):", adjusted_rand_score(y[:len(p)], p))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y[:len(p)]<0, s))

 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Nur für Visualisierung: LabelEncoder auf nicht-Outlier anwenden
p_vis = p.copy()
if np.sum(p_vis > -1) > 0:
    p_vis[p_vis > -1] = LabelEncoder().fit_transform(p_vis[p_vis > -1])

fig = plt.figure(figsize=(15,4))
cmap = plt.get_cmap('tab20', len(np.unique(p_vis[p_vis > -1])) if np.sum(p_vis > -1) > 0 else 1)

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    if np.sum(p_vis > -1) > 0:
        ax.scatter3D(t[p_vis > -1], x[p_vis > -1, 0], x[p_vis > -1, 1], s=5, c=p_vis[p_vis > -1], cmap=cmap)
    if np.sum(p_vis == -1) > 0:
        ax.scatter3D(t[p_vis == -1], x[p_vis == -1, 0], x[p_vis == -1, 1], s=5, c='black')
    ax.view_init(azim=280+i*30, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.tight_layout()
plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
print("\nVisualisierung gespeichert als 'demo_output.png'")
plt.close()

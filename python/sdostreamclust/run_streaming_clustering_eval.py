#!/usr/bin/env python3
"""
Pure Streaming Clustering Evaluation Framework
Focus on SDOstreamclust with no warmup phase, synthetic datasets, ARI evaluation only
No competitor algorithms - only Ground Truth comparison
"""

import sys
import os
import glob
import re
import time
import numpy as np
import pandas as pd
from scipy.io import arff

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

try:
    from sdo import SDOstreamclust
    print("SDOstreamclust module imported successfully")
except ImportError as e:
    print(f"Error: Could not import sdo module: {e}")
    print("Please install the module with 'maturin develop' or 'pip install .'")
    sys.exit(1)

# Try to import sklearn with proper error handling
try:
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics.cluster import adjusted_rand_score
    print("sklearn imported successfully")
except ImportError as e:
    print(f"Error: Could not import sklearn: {e}")
    print("Please install sklearn: pip install scikit-learn")
    sys.exit(1)

def discover_datasets(folders):
    """Discover all ARFF datasets in specified folders"""
    datasets = []
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: Dataset folder {folder} does not exist")
            continue
            
        arff_files = glob.glob(os.path.join(folder, '*.arff'))
        arff_files.sort()
        
        for arff_file in arff_files:
            dataset_info = {
                'filepath': arff_file,
                'filename': os.path.basename(arff_file),
                'folder': os.path.basename(os.path.dirname(arff_file)),
                'full_path': arff_file,
                'category': folder
            }
            datasets.append(dataset_info)
    
    print(f"Discovered {len(datasets)} datasets in folders: {folders}")
    return datasets

def load_streaming_data(filename):
    """Load entire dataset as streaming data (no train/test split)"""
    print(f"Loading dataset: {filename}")
    
    # Load ARFF file
    with open(filename, 'r') as f:
        arff_data = arff.loadarff(f)
    
    df_data = pd.DataFrame(arff_data[0])
    
    # Store original class values for ARI calculation
    original_class_values = df_data['class'].copy()
    
    # Handle class column encoding
    if 'class' in df_data.columns:
        if df_data['class'].dtypes == 'object':
            df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip('') if isinstance(x, bytes) else x)
        
        # Check for outlier labels and handle them properly
        unique_classes = df_data['class'].unique()
        has_outliers = '-1' in unique_classes
        
        if has_outliers:
            # Simple approach: replace -1 with 'outlier' temporarily, encode, then handle separately
            df_temp = df_data.copy()
            outlier_mask = df_temp['class'] == '-1'
            df_temp.loc[~outlier_mask, 'class'] = 'cluster_' + df_temp.loc[~outlier_mask, 'class']
            df_temp.loc[outlier_mask, 'class'] = 'outlier'
            
            # Encode all labels now
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df_temp['class'])
            
            # Get original mappings including outlier
            true_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
        else:
            # Normal case: no outliers
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df_data['class'])
            true_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
    else:
        print(f"Warning: No 'class' column found in {filename}")
        return None, None, None, None, None, None
    
    # Remove class column, keep only features
    df_data.drop(columns=['class'], inplace=True)
    x = df_data.to_numpy()
    
    # Normalize features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Create streaming iterator
    def stream_iterator():
        for i in range(len(x_scaled)):
            yield x_scaled[i], y[i]
    
    n_samples, n_features = x_scaled.shape
    n_classes = len(label_encoder.classes_)
    
    # Handle outliers counting properly
    if has_outliers:
        outliers = np.sum(original_class_values == '-1')
    else:
        outliers = 0
    
    dataset_info = {
        'filename': os.path.basename(filename),
        'folder': os.path.basename(os.path.dirname(filename)),
        'x': x_scaled,
        'y': y,
        'true_labels': true_labels,
        'original_class_values': original_class_values,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_outliers': outliers,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    
    print(f"  Shape: {n_samples} samples × {n_features} features")
    print(f"  Classes: {n_classes} (including outliers)")
    print(f"  Outliers: {outliers}")
    
    return stream_iterator(), dataset_info

def evaluate_no_warmup_streaming(dataset_info, params):
    """Evaluate SDOstreamclust with no warmup phase"""
    print(f"\n=== Evaluating {dataset_info['filename']} ===")
    print(f"Parameters: {params}")
    
    # Initialize SDOstreamclust with dimension only (no warmup)
    model = SDOstreamclust(
        k=params['k'],
        x=params['x'], 
        t_fading=params['t_fading'],
        chi_min=params['chi_min'],
        chi_prop=params['chi_prop'], 
        zeta=params['zeta'],
        min_cluster_size=params['min_cluster_size'],
        dimension=dataset_info['n_features']
    )
    
    print(f"Initialized with {params['k']} observers for {dataset_info['n_features']} dimensions")
    
    # Process entire dataset as stream (no warmup)
    start_time = time.time()
    predictions = []
    true_labels_used = []
    
    data_stream = dataset_info['x']
    original_class_values = dataset_info['original_class_values']
    true_label_mapping = dataset_info['true_labels']
    
    for i, (point, true_label_idx) in enumerate(zip(data_stream, dataset_info['y'])):
        # Process point
        point_2d = np.array([point], dtype=np.float64)
        
        try:
            label, score = model.learn(point_2d)
            predictions.append(label)
            
            # Map encoded prediction back to original class for ARI
            if 0 <= label < len(true_label_mapping):
                true_labels_used.append(true_label_mapping[label])
            else:
                true_labels_used.append(-1)  # Outlier case
                
        except Exception as e:
            print(f"Error processing point {i}: {e}")
            predictions.append(-1)
            # Use original class value directly
            if i < len(original_class_values):
                true_labels_used.append(original_class_values[i])
            else:
                true_labels_used.append(-1)
        
        # Progress indicator every 500 points
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data_stream)} points...")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate final ARI
    ari = adjusted_rand_score(true_labels_used, predictions)
    
    results = {
        'filename': dataset_info['filename'],
        'dataset_category': dataset_info['folder'],
        'algorithm': 'SDOstreamclust_NO_WARMUP',
        'k': params['k'],
        'x': params['x'],
        't_fading': params['t_fading'],
        'chi_min': params['chi_min'],
        'chi_prop': params['chi_prop'],
        'zeta': params['zeta'],
        'min_cluster_size': params['min_cluster_size'],
        'n_samples': dataset_info['n_samples'],
        'n_features': dataset_info['n_features'],
        'n_classes': dataset_info['n_classes'],
        'n_outliers': dataset_info['n_outliers'],
        'ARI': ari,
        'processing_time': processing_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"  Final ARI: {ari:.4f}")
    print(f"  Processing time: {processing_time:.2f} seconds")
    
    return results

def save_results(results, output_path, log_level='INFO'):
    """Save results to CSV file"""
    # Define CSV columns
    columns = [
        'filename', 'dataset_category', 'algorithm', 'k', 'x', 't_fading', 
        'chi_min', 'chi_prop', 'zeta', 'min_cluster_size',
        'n_samples', 'n_features', 'n_classes', 'n_outliers',
        'ARI', 'processing_time', 'timestamp'
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_path, 'streaming_results.csv')
    
    if os.path.exists(output_file):
        # Load existing results
        existing_df = pd.read_csv(output_file)
        new_results_df = pd.DataFrame([results])
        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Results appended to {output_file}")
    else:
        # Create new results file
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print(f"\n=== Result Summary ===")
    print(f"Dataset: {results['filename']}")
    print(f"Category: {results['dataset_category']}")
    print(f"ARI: {results['ARI']:.4f}")
    print(f"Processing time: {results['processing_time']:.2f}s")
    print(f"Saved to: {output_file}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pure Streaming Clustering Evaluation')
    parser.add_argument('--data-folders', nargs='+', 
                       default=[os.path.join(_REPO_ROOT, 'evaluation_tests', 'data', 'synthetic')],
                       help='Dataset folders to process')
    parser.add_argument('--output', 
                       default=os.path.join(_THIS_DIR, 'out'),
                       help='Output directory for results (default: python/sdostreamclust/out)')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pure Streaming Clustering Evaluation")
    print("SDOstreamclust - No Warmup Phase")
    print("=" * 60)
    print(f"Data folders: {args.data_folders}")
    print(f"Output directory: {args.output}")
    print(f"Log level: {args.log_level}")
    print()
    
    # Default SDOstreamclust parameters (no warmup scenario)
    default_params = {
        'k': 50,                # Number of observers
        'x': 5,                 # Nearest neighbors for scoring
        't_fading': 20.0,       # Fading parameter
        'chi_min': 1,            # Minimum neighbors for local threshold
        'chi_prop': 0.1,         # Proportional chi (10% of k)
        'zeta': 0.6,             # Global/local mixing parameter
        'min_cluster_size': 2      # Minimum cluster size
    }
    
    print(f"Default parameters: {default_params}")
    print()
    
    # Discover datasets
    datasets = discover_datasets(args.data_folders)
    
    if not datasets:
        print("No datasets found!")
        return
    
    # For now: process first dataset only
    dataset = datasets[0]  # Start with first dataset found
    
    print(f"Selected dataset: {dataset['filename']}")
    
    # Load streaming data
    stream_data, dataset_info = load_streaming_data(dataset['full_path'])
    
    if stream_data is None:
        print(f"Failed to load dataset: {dataset['filename']}")
        return
    
    # Evaluate streaming clustering
    results = evaluate_no_warmup_streaming(dataset_info, default_params)
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n=== Evaluation Complete ===")
    print("Results saved successfully!")

if __name__ == "__main__":
    main()
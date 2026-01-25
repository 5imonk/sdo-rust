#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sdo import SDOstream
import os
import glob
import sys

def evaluate_file(filename, k=200, x=5, t_fading=100.0, rho=0.1):
    print(f"Evaluating: {filename}")
    
    # Load ARFF
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)
    
    # Identify class column
    class_col = 'class'
    if class_col not in df.columns:
        # Try to find the last column if 'class' is not present
        class_col = df.columns[-1]
    
    # Convert class to labels
    # Outliers are labeled as 0
    y_raw = df[class_col]
    if y_raw.dtype == object or y_raw.dtype.name == 'bytes':
        y_raw = y_raw.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    
    # y = 1 for outlier (label 0), 0 otherwise
    # The user says: "outliers are categorized with label 0"
    # Convert labels to strings or ints as they appear in the ARFF
    try:
        y = (y_raw.astype(int) == 0).astype(int)
    except ValueError:
        # If it's nominal like '0', '1', etc.
        y = (y_raw.astype(str) == '0').astype(int)
        
    print(f"Points: {len(y)}, Outliers: {np.sum(y)}")
    
    # Prepare features
    X_df = df.drop(columns=[class_col])
    X = X_df.to_numpy().astype(np.float64)
    
    # Normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Initialize SDOstream
    # Using dimension for initialization
    classifier = SDOstream(
        k=k, 
        x=x, 
        t_fading=t_fading, 
        t_sampling=t_fading, 
        rho=rho, 
        dimension=X.shape[1]
    )
    
    scores = []
    
    # Iterate and learn
    for i in range(len(X)):
        # Reshape to (1, dim) for PyReadonlyArray2
        point = X[i:i+1]
        # learn returns the median distance (outlier score)
        score = classifier.learn(point)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Calculate ROC-AUC
    try:
        auc = roc_auc_score(y, scores)
    except ValueError as e:
        print(f"Error calculating AUC for {filename}: {e}")
        return None
        
    print(f"ROC-AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "evaluation_tests/data/outlier/base/arff/base_data_1.arff"
    
    if os.path.isfile(target):
        evaluate_file(target)
    elif os.path.isdir(target):
        files = glob.glob(os.path.join(target, "**/*.arff"), recursive=True)
        results = {}
        for f in sorted(files):
            auc = evaluate_file(f)
            if auc is not None:
                results[f] = auc
        
        if results:
            print("\nSummary Results:")
            for f, auc in results.items():
                print(f"{os.path.basename(f)}: {auc:.4f}")
            print(f"Average ROC-AUC: {np.mean(list(results.values())):.4f}")
    else:
        print(f"Target {target} not found.")

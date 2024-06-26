import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import h5py

# Load Data
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    return data

train_inputs = load_data('data/train_multi_inputs.h5')
train_targets = load_data('data/train_multi_targets.h5')
test_inputs = load_data('data/test_multi_inputs.h5')

# Normalize Data
scaler = StandardScaler()
train_inputs_normalized = scaler.fit_transform(train_inputs)
test_inputs_normalized = scaler.transform(test_inputs)

# Dimensionality Reduction
pca = PCA(n_components=100)
train_inputs_pca = pca.fit_transform(train_inputs_normalized)
test_inputs_pca = pca.transform(test_inputs_normalized)

# Impute Missing Values
imputer = KNNImputer(n_neighbors=5)
train_inputs_imputed = imputer.fit_transform(train_inputs_pca)
test_inputs_imputed = imputer.transform(test_inputs_pca)

# Save Preprocessed Data
np.save('data/train_inputs_preprocessed.npy', train_inputs_imputed)
np.save('data/test_inputs_preprocessed.npy', test_inputs_imputed)
np.save('data/train_targets.npy', train_targets)

print("Data preprocessing complete.")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import h5py

# Function to load data from HDF5 files
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    return data

# Load raw data
train_inputs = load_data('data/train_multi_inputs.h5')
train_targets = load_data('data/train_multi_targets.h5')
test_inputs = load_data('data/test_multi_inputs.h5')

# Normalize data
scaler = StandardScaler()
train_inputs_normalized = scaler.fit_transform(train_inputs)
test_inputs_normalized = scaler.transform(test_inputs)

# Dimensionality reduction
pca = PCA(n_components=100)
train_inputs_pca = pca.fit_transform(train_inputs_normalized)
test_inputs_pca = pca.transform(test_inputs_normalized)

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
train_inputs_imputed = imputer.fit_transform(train_inputs_pca)
test_inputs_imputed = imputer.transform(test_inputs_pca)

# Save preprocessed data
np.savetxt('data/train_inputs_preprocessed.py', train_inputs_imputed)
np.savetxt('data/test_inputs_preprocessed.py', test_inputs_imputed)
np.savetxt('data/train_targets.py', train_targets)

print("Data preprocessing complete.")
----------------------------------------------------------------------------------------------------------------------------------------
# Explanation
# Load Data:
The load_data function reads data from the HDF5 files and returns it as NumPy arrays.
train_inputs, train_targets, and test_inputs are loaded from their respective HDF5 files.
# Normalize Data:
The StandardScaler is used to normalize the features. This standardization step ensures that each feature has a mean of 0 and a standard deviation of 1.
# Dimensionality Reduction:
The PCA (Principal Component Analysis) is applied to reduce the dimensionality of the data to 100 components. This can help reduce the complexity and improve model performance.
# Impute Missing Values:
The KNNImputer is used to fill in any missing values in the dataset. It uses the mean of the nearest neighbors to impute missing values.
# Save Preprocessed Data:
The preprocessed training inputs, test inputs, and training targets are saved as NumPy arrays in the data directory.
Running the Code
Save the script as data_preprocessing.py in the src directory and run it from the terminal: python src/data_preprocessing.py

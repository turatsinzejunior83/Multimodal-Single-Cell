import numpy as np
import h5py
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

# Configure logging
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path, dataset_name='data'):
    """ Load data from HDF5 file """
    try:
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:]
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def preprocess_data(train_inputs, test_inputs):
    """ Preprocess data by normalizing, reducing dimensionality, and imputing missing values """
    try:
        # Normalize data
        scaler = StandardScaler()
        train_inputs_normalized = scaler.fit_transform(train_inputs)
        test_inputs_normalized = scaler.transform(test_inputs)
        logging.info("Data normalization complete.")
        
        # Dimensionality reduction
        pca = PCA(n_components=100)
        train_inputs_pca = pca.fit_transform(train_inputs_normalized)
        test_inputs_pca = pca.transform(test_inputs_normalized)
        logging.info("Dimensionality reduction complete.")
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        train_inputs_imputed = imputer.fit_transform(train_inputs_pca)
        test_inputs_imputed = imputer.transform(test_inputs_pca)
        logging.info("Missing value imputation complete.")
        
        return train_inputs_imputed, test_inputs_imputed
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise

def save_preprocessed_data(data, file_path):
    """ Save preprocessed data to a NumPy file """
    try:
        np.save(file_path, data)
        logging.info(f"Successfully saved preprocessed data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data to {file_path}: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load raw data
        train_inputs = load_data('data/train_multi_inputs.h5')
        train_targets = load_data('data/train_multi_targets.h5')
        test_inputs = load_data('data/test_multi_inputs.h5')
        
        # Preprocess data
        train_inputs_preprocessed, test_inputs_preprocessed = preprocess_data(train_inputs, test_inputs)
        
        # Save preprocessed data
        save_preprocessed_data(train_inputs_preprocessed, 'data/train_inputs_preprocessed.npy')
        save_preprocessed_data(test_inputs_preprocessed, 'data/test_inputs_preprocessed.npy')
        save_preprocessed_data(train_targets, 'data/train_targets.npy')
        
        logging.info("Data preprocessing complete.")
    except Exception as e:
        logging.critical(f"Critical error in the data preprocessing pipeline: {e}")

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

---------------------------------------------------------------------------------------------------------------------------
#Detailed Breakdown of Enhancements
1. Logging Configuration
The script is configured to log messages to a file named data_preprocessing.log. The logging format includes timestamps, log levels, and messages for better traceability.
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
2. Loading Data with Error Handling
The load_data function includes error handling to log and raise exceptions if any issues occur during data loading.
def load_data(file_path, dataset_name='data'):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:]
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise
3. Data Preprocessing with Error Handling
The preprocess_data function takes care of normalizing, reducing dimensionality, and imputing missing values. It logs the completion of each preprocessing step and raises exceptions if any errors occur.
def preprocess_data(train_inputs, test_inputs):
   def preprocess_data(train_inputs, test_inputs):
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
    # Normalization:
#Uses StandardScaler to standardize features by removing the mean and scaling to unit variance.
#fit_transform is applied to the training data to learn the parameters and transform the data.
#transform is applied to the test data using the parameters learned from the training data to ensure consistency.
  # Dimensionality Reduction:
#Uses PCA to reduce the number of features to 100 principal components.
#fit_transform is used on the training data to learn the principal components and transform the data.
#transform is used on the test data to transform it using the learned principal components from the training data.
   # Imputation:
#Uses KNNImputer to fill in missing values by averaging the values of the nearest neighbors.
#fit_transform is applied to the training data to learn the imputation and apply it.
#transform is used on the test data to apply the same imputation strategy.
4. Saving Preprocessed Data with Error Handling
#The save_preprocessed_data function saves NumPy arrays to disk and logs the process. It includes error handling to catch and log any issues during the save operation.
def save_preprocessed_data(data, file_path):
    try:
        np.save(file_path, data)
        logging.info(f"Successfully saved preprocessed data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data to {file_path}: {e}")
        raise
#NumPy Save:
np.save is used to save the data to a .npy file.
The function logs the success or failure of the save operation.
Main Script Execution
The main part of the script orchestrates the loading, preprocessing, and saving of data. It includes comprehensive logging and error handling to ensure the entire pipeline is robust.
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

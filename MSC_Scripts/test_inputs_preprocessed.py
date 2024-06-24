import os
import numpy as np

def create_directory(directory_path):
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)

def generate_train_targets(number_of_samples, number_of_targets, file_path, seed=None, is_classification=False):
    """Generate and save training targets."""
    if seed is not None:
        np.random.seed(seed)
    
    if is_classification:
        # Generate random binary data for classification targets
        train_targets = np.random.randint(0, 2, size=(number_of_samples, number_of_targets))
    else:
        # Generate random data for regression targets
        train_targets = np.random.rand(number_of_samples, number_of_targets)
    
    # Save the array to a .py file
    np.savetxt(file_path, train_targets)
    print(f"Generated and saved {file_path}")

# Define the parameters
number_of_samples = 1000
number_of_targets = 10
directory_path = 'data'
file_path = os.path.join(directory_path, 'train_targets.py')

# Ensure the directory exists
create_directory(directory_path)

# Generate and save the data
generate_train_targets(number_of_samples, number_of_targets, file_path, seed=42, is_classification=False)

----------------------------------------------------------------
# Explanation
NumPy Library: Both scripts use the NumPy library to create random arrays and save them as .npy files.
Random Data Generation: np.random.rand generates arrays with random floats between 0 and 1. Adjust the range and distribution if necessary to better match your actual data.
File Saving: np.save saves the generated arrays to the specified file paths.
1. Running the Scripts
Set Up the Environment: Ensure you have NumPy installed. You can install it using pip if necessary:
pip install numpy
2. Run the Scripts: Execute the scripts in your Python environment:
python generate_test_inputs_preprocessed.py
python generate_train_targets.py
These scripts will create and save the .npy files in the data directory, ready for use in your preprocessing and modeling pipeline. Adjust the number of samples and features/targets as needed to match your specific use case.


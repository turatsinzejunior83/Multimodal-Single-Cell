import os
import numpy as np

def create_directory(directory_path):
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)

def generate_test_inputs_preprocessed(number_of_samples, number_of_features, file_path, seed=None):
    """Generate and save preprocessed test inputs."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random data for test inputs using a normal distribution
    test_inputs_preprocessed = np.random.normal(loc=0.0, scale=1.0, size=(number_of_samples, number_of_features))
    
    # Save the array to a .npy file
    np.save(file_path, test_inputs_preprocessed)
    print(f"Generated and saved {file_path}")

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
    
    # Save the array to a .npy file
    np.save(file_path, train_targets)
    print(f"Generated and saved {file_path}")

# Define the parameters
number_of_samples = 1000
number_of_features = 100
number_of_targets = 10
directory_path = 'data'

# Ensure the directory exists
create_directory(directory_path)

# Generate and save the test inputs
generate_test_inputs_preprocessed(
    number_of_samples=number_of_samples, 
    number_of_features=number_of_features, 
    file_path=os.path.join(directory_path, 'test_inputs_preprocessed.npy'), 
    seed=42
)

# Generate and save the train targets
generate_train_targets(
    number_of_samples=number_of_samples, 
    number_of_targets=number_of_targets, 
    file_path=os.path.join(directory_path, 'train_targets.npy'), 
    seed=42, 
    is_classification=False  # Set to True if your targets are for classification
)
  
# Explanation
Creating Directories: The create_directory function ensures that the data directory exists before attempting to save any files.
Generating Test Inputs: generate_test_inputs_preprocessed creates a NumPy array with random values from a normal distribution and saves it to test_inputs_preprocessed.npy.
Generating Train Targets: generate_train_targets creates a NumPy array with either random binary values (for classification) or random float values (for regression) and saves it to train_targets.npy.
Running the Script
You can run this script in your Python environment. Make sure you have NumPy installed:pip install numpy
Executing the script will generate two .npy files in the data directory: test_inputs_preprocessed.npy and train_targets.npy. Adjust the parameters as needed to fit your specific requirements.

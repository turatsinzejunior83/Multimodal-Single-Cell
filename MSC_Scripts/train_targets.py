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

import os
import numpy as np
import joblib

def create_directory(directory_path):
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)

# Directory to save the predictions
predictions_directory = 'predictions'
create_directory(predictions_directory)

# Define the file path for the model and the predictions
model_file_path = 'models/random_forest.pkl'
predictions_file_path = os.path.join(predictions_directory, 'rf_preds.py')

# Generate some random data for making predictions
np.random.seed(42)
X_test = np.random.rand(20, 10)  # 20 samples, 10 features (adjust as needed)

# Load the trained Random Forest model
rf_model = joblib.load(model_file_path)

# Make predictions using the model
predictions = rf_model.predict(X_test)

# Save the predictions to a .py file
np.savetxt(predictions_file_path, predictions)
print(f"Generated and saved {predictions_file_path}")

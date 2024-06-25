import os
import numpy as np
from tensorflow.keras.models import load_model

def create_directory(directory_path):
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)

# Directory to save the predictions
predictions_directory = 'predictions'
create_directory(predictions_directory)

# Define the file path for the model and the predictions
model_file_path = 'models/transformer_model.h5'
predictions_file_path = os.path.join(predictions_directory, 'transformer_preds.py')

# Generate some random data for making predictions
np.random.seed(42)
X_test = np.random.rand(20, 10, 10)  # 20 samples, 10x10 features (adjust as needed)

# Load the trained Transformer model
transformer_model = load_model(model_file_path)

# Make predictions using the model
predictions = transformer_model.predict(X_test)

# Save the predictions to a .py file
np.savetxt(predictions_file_path, predictions)
print(f"Generated and saved {predictions_file_path}")

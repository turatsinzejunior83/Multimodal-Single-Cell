import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import r2_score

# Load Preprocessed Data
test_inputs = np.load('data/test_inputs_preprocessed.npy')

# Load Models
lr = joblib.load('models/linear_regression.pkl')
rf = joblib.load('models/random_forest.pkl')
lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
transformer_model = tf.keras.models.load_model('models/transformer_model.h5')

# Reshape test data for LSTM and Transformer
test_inputs_lstm = test_inputs.reshape((test_inputs.shape[0], test_inputs.shape[1], 1))

# Predict with Linear Regression
lr_preds = lr.predict(test_inputs)

# Predict with Random Forest
rf_preds = rf.predict(test_inputs)

# Predict with LSTM
lstm_preds = lstm_model.predict(test_inputs_lstm)

# Predict with Transformer
transformer_preds = transformer_model.predict(test_inputs_lstm)

# Here, you would typically have ground truth for the test set to calculate R^2
# Since we are assuming the test set is unseen, we will skip R^2 calculation for test predictions
# However, for demonstration, let's assume we have some dummy ground truth
dummy_ground_truth = np.random.rand(test_inputs.shape[0], lr_preds.shape[1])

# Evaluate predictions (this is just for demonstration)
lr_r2_test = r2_score(dummy_ground_truth, lr_preds)
rf_r2_test = r2_score(dummy_ground_truth, rf_preds)
lstm_r2_test = r2_score(dummy_ground_truth, lstm_preds)
transformer_r2_test = r2_score(dummy_ground_truth, transformer_preds)

print(f'Linear Regression Test R^2 Score: {lr_r2_test}')
print(f'Random Forest Test R^2 Score: {rf_r2_test}')
print(f'LSTM Test R^2 Score: {lstm_r2_test}')
print(f'Transformer Test R^2 Score: {transformer_r2_test}')

# Save Predictions
np.save('predictions/lr_preds.npy', lr_preds)
np.save('predictions/rf_preds.npy', rf_preds)
np.save('predictions/lstm_preds.npy', lstm_preds)
np.save('predictions/transformer_preds.npy', transformer_preds)

print("Evaluation complete.")

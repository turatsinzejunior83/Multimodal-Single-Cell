import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load Preprocessed Data
train_inputs = np.load('data/train_inputs_preprocessed.npy')
train_targets = np.load('data/train_targets.npy')

# Split Data
X_train, X_val, y_train, y_val = train_test_split(train_inputs, train_targets, test_size=0.2, random_state=42)

# Reshape for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Define LSTM Model
lstm_model = Sequential([
    LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1])
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# Train LSTM Model
history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_val_lstm, y_val))

# Evaluate LSTM Model
lstm_preds = lstm_model.predict(X_val_lstm)
lstm_r2 = r2_score(y_val, lstm_preds)
print(f'LSTM R^2 Score: {lstm_r2}')

# Save LSTM model
lstm_model.save('models/lstm_model.h5')

# Transformer-based Model (Optional)
# Note: This is a simplified example. For a full implementation, refer to transformers library.
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Define Transformer Model
input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
inputs = tf.keras.Input(shape=input_shape)
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(y_train.shape[1])(x)

transformer_model = tf.keras.Model(inputs, outputs)
transformer_model.compile(optimizer='adam', loss='mse')
transformer_model.summary()

# Train Transformer Model
history = transformer_model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_val_lstm, y_val))

# Evaluate Transformer Model
transformer_preds = transformer_model.predict(X_val_lstm)
transformer_r2 = r2_score(y_val, transformer_preds)
print(f'Transformer R^2 Score: {transformer_r2}')

# Save Transformer model
transformer_model.save('models/transformer_model.h5')

print("Advanced model training complete.")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load Preprocessed Data
train_inputs = np.load('data/train_inputs_preprocessed.npy')
train_targets = np.load('data/train_targets.npy')

# Split Data
X_train, X_val, y_train, y_val = train_test_split(train_inputs, train_targets, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_val)
lr_r2 = r2_score(y_val, lr_preds)
print(f'Linear Regression R^2 Score: {lr_r2}')

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)
rf_r2 = r2_score(y_val, rf_preds)
print(f'Random Forest R^2 Score: {rf_r2}')

# Save models (optional)
import joblib
joblib.dump(lr, 'models/linear_regression.pkl')
joblib.dump(rf, 'models/random_forest.pkl')

print("Baseline model training complete.")

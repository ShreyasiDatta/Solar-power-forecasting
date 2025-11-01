import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Bidirectional, Dense,
    Conv1D, MaxPooling1D, Flatten,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#################################################################
### 1. HELPER FUNCTIONS
#################################################################

def create_sequences(data, lookback, horizon, target_col_index):
    """
    Creates sequences of data for time series forecasting.
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + horizon - 1, target_col_index])
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns a dictionary of evaluation metrics.
    Handles MAPE's divide-by-zero issue.
    """
    # Filter out zero values from y_true for MAPE calculation
    non_zero_mask = y_true > 1e-3  # Use a small threshold to avoid precision issues
    
    if np.sum(non_zero_mask) == 0:
        mape = np.nan # or 0, depending on how you want to handle all-zero actuals
    else:
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])

    metrics = {
        'RMSE (MW)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE (MW)': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': mape * 100
    }
    return metrics

#################################################################
### 2. SIMULATE AND PREPROCESS DATA
#################################################################
# --- !!! REPLACE THIS SECTION WITH YOUR OWN DATA LOADING !!! ---

print("Step 2: Simulating and preprocessing data...")

# Create 1 year of hourly data
n_samples = 365 * 24
time_index = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')

# line 67:
# Convert to numpy arrays to ensure calculations result in a mutable array
hour_of_day = time_index.hour.to_numpy()
day_of_year = time_index.dayofyear.to_numpy()

# Daily cycle (0 at night, peaks at noon)
daily_cycle = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * 100)
# Seasonal cycle (stronger in summer)
seasonal_cycle = (1 + 0.5 * np.sin((day_of_year - 80) * 2 * np.pi / 365.25))
# Noise
noise = np.random.rand(n_samples) * 10

# Combine
power = daily_cycle * seasonal_cycle + noise
power[power < 0] = 0 # Ensure no negative power

df = pd.DataFrame(data={'power': power}, index=time_index)

# --- END OF DATA SIMULATION ---

# --- Feature Engineering (Cyclical Features) ---
df['hour'] = df.index.hour
df['day_of_year'] = df.index.dayofyear

# Add cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Define the features to be used by the model
# 'power' MUST be the first column for the inverse_transform to work later
feature_columns = ['power', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
features_df = df[feature_columns]

# --- Train/Test Split (CHRONOLOGICAL) ---
split_index = int(len(features_df) * 0.8)
train_df = features_df.iloc[:split_index]
test_df = features_df.iloc[split_index:]

# --- Scaling ---
# FIT the scaler ONLY on the training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
# TRANSFORM the test data using the *same* scaler
test_scaled = scaler.transform(test_df)

#################################################################
### 3. CREATE TIME-SERIES SEQUENCES
#################################################################
print("Step 3: Creating time-series sequences...")

LOOKBACK = 72  # 72 hours of history
HORIZON = 1    # 1 hour into the future
TARGET_COL_INDEX = 0 # 'power' is the first column (index 0)

X_train, y_train = create_sequences(train_scaled, LOOKBACK, HORIZON, TARGET_COL_INDEX)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, HORIZON, TARGET_COL_INDEX)

n_features = X_train.shape[2]
input_shape = (LOOKBACK, n_features)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#################################################################
### 4. DEFINE MODEL ARCHITECTURES
#################################################################
print("Step 4: Defining model architectures...")

def build_model(model_name, input_shape):
    """
    Builds a Keras model based on the specified model name.
    """
    inputs = Input(shape=input_shape)
    
    # --- RNN Architectures ---
    if model_name == 'lstm':
        x = LSTM(64, activation='relu')(inputs)
    
    elif model_name == 'gru':
        x = GRU(64, activation='relu')(inputs)
        
    elif model_name == 'bilstm':
        x = Bidirectional(LSTM(64, activation='relu'))(inputs)
        
    # --- Hybrid Architecture ---
    elif model_name == 'cnn_lstm':
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(64, activation='relu')(x)
        
    # --- Attention Architecture ---
    elif model_name == 'transformer':
        # A minimal Transformer Encoder block
        num_heads = 4
        key_dim = 64
        
        # Self-Attention
        x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x + inputs) # Add & Norm
        
        # Feed-Forward
        ff_input = x
        x = Dense(128, activation='relu')(x)
        x = Dense(input_shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x + ff_input) # Add & Norm
        
        # To single output
        x = GlobalAveragePooling1D()(x)
        
    else:
        raise ValueError("Unknown model name")

    # --- Output Layer ---
    # Final prediction. 'linear' activation for regression.
    outputs = Dense(1, activation='linear')(x) 
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error
    )
    return model

#################################################################
### 5. TRAIN AND EVALUATE MODELS
#################################################################
print("Step 5: Training and evaluating models...")

models_to_run = ['lstm', 'gru', 'bilstm', 'cnn_lstm', 'transformer']
evaluation_results = {}

# Use EarlyStopping to prevent overfitting and speed up training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for model_name in models_to_run:
    print(f"\n--- Training Model: {model_name.upper()} ---")
    
    # Build the model
    model = build_model(model_name, input_shape)
    
    if model_name == 'transformer':
        print("Note: Transformer model may take longer to train...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20, # Increase this for better performance (e.g., 50-100)
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- Evaluate ---
    # Predict on the test set
    y_pred_scaled = model.predict(X_test)
    
    # --- CRITICAL: Inverse Transform Predictions ---
    # We must "un-scale" the predictions and true values to get meaningful MW errors
    
    # Reshape y_test to be (n_samples, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # Create "dummy" arrays with the same shape as the scaler expects (n_samples, n_features)
    # Put our predicted value in the first column (which was 'power')
    dummy_pred = np.zeros((len(y_pred_scaled), n_features))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    
    dummy_test = np.zeros((len(y_test_reshaped), n_features))
    dummy_test[:, 0] = y_test_reshaped.flatten()

    # Now, inverse transform
    y_pred_unscaled = scaler.inverse_transform(dummy_pred)[:, 0]
    y_test_unscaled = scaler.inverse_transform(dummy_test)[:, 0]

    # Ensure no negative predictions (solar power can't be negative)
    y_pred_unscaled[y_pred_unscaled < 0] = 0
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_unscaled, y_pred_unscaled)
    evaluation_results[model_name] = metrics
    
    print(f"--- Results for {model_name.upper()} ---")
    print(metrics)


#################################################################
### 6. COMPARE RESULTS
#################################################################
print("\n--- Final Model Comparison ---")

results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
results_df = results_df.sort_values(by='RMSE (MW)') # Sort by best RMSE

print(results_df.to_markdown(floatfmt=".3f"))
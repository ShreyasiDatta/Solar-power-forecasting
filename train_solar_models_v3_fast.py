# ...existing code...
import os
import sys
import numpy as np
import pandas as pd
import time
import shutil

# --- Core ML ---
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Bidirectional, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten, Embedding,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- KerasTuner (Fixed Imports) ---
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch

# --- Evaluation & Explainability ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 0. Project Configuration ---
RESULTS_DIR = "project_results_v3_fast"
DATA_FILE = 'time_series_60min_singleindex.csv'
TUNER_DIR = "kerastuner_logs_fast"

LOOKBACK = 72  # 72 hours of history
HORIZON = 1    # 1 hour into the future
BATCH_SIZE = 64  # sensible default to speed training/prediction

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


#################################################################
### 1. Visualization & Utility Functions
#################################################################

def create_sequences(data, lookback, horizon, target_col_index):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + horizon - 1, target_col_index])
    return np.array(X), np.array(y)


def calculate_metrics(y_true, y_pred):
    non_zero_mask = y_true > 1e-3
    mape = np.nan
    if np.sum(non_zero_mask) > 0:
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    metrics = {
        'RMSE (MW)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE (MW)': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': mape * 100
    }
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir):
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.3, label="Predictions")
    lims = [np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])]
    plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="Perfect Prediction")
    plt.xlabel('Actual Power (MW)')
    plt.ylabel('Predicted Power (MW)')
    plt.title(f'{model_name.upper()} - Actual vs. Predicted (Holdout Test Set)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    save_path = os.path.join(save_dir, "plot_actual_vs_predicted.png")
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name, save_dir):
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title(f'{model_name.upper()} - Error Distribution (Holdout Test Set)')
    plt.xlabel('Prediction Error (Actual - Predicted) (MW)')
    plt.ylabel('Frequency')
    plt.axvline(x=errors.mean(), color='red', linestyle='--', label=f'Mean Error: {errors.mean():.2f}')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "plot_error_distribution.png")
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results_df, metric, save_dir):
    plt.figure(figsize=(12, 7))
    results_df = results_df.sort_values(by=metric, ascending=True)
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    save_path = os.path.join(save_dir, f"plot_model_comparison_{metric.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_shap_summary(shap_values, features, feature_names, model_name, save_dir):
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values_reshaped = shap_values.reshape(-1, shap_values.shape[-1])
    features_reshaped = features.reshape(-1, features.shape[-1])
    shap.summary_plot(shap_values_reshaped, features_reshaped, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance for {model_name.upper()}')
    save_path = os.path.join(save_dir, "plot_shap_summary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


#################################################################
### 2. Load Real Data (FIXED Columns)
#################################################################
print(f"Step 2: Loading and preprocessing data from {DATA_FILE}...")

feature_cols_to_load = [
    'utc_timestamp',
    'DE_solar_generation_actual',
    'DE_load_actual_entsoe_transparency'
]

try:
    df = pd.read_csv(
        DATA_FILE,
        usecols=feature_cols_to_load,
        parse_dates=['utc_timestamp'],
        index_col='utc_timestamp'
    )
except FileNotFoundError:
    print(f"---! ERROR: File not found: {DATA_FILE} !---")
    print("Please download it from 'https://data.open-power-system-data.org/time_series/'")
    sys.exit()
except ValueError as e:
    print(f"---! ERROR: Missing expected columns in CSV. !---")
    print(f"Full error: {e}")
    sys.exit()

df.rename(columns={
    'DE_solar_generation_actual': 'power',
    'DE_load_actual_entsoe_transparency': 'load'
}, inplace=True)

df.sort_index(inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df['power'] = df['power'].clip(lower=0)

df['hour'] = df.index.hour
df['day_of_year'] = df.index.dayofyear
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

feature_columns = [
    'power', 'load',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]
features_df = df[feature_columns]
TARGET_COL_INDEX = 0
N_FEATURES = len(feature_columns)

print(f"Loaded {len(df)} records. Using {N_FEATURES} features.")


#################################################################
### 3. NEW: Proper 3-Way Data Split (Train / Val / Test)
#################################################################
print("Step 3: Creating 70/15/15 Train/Validation/Test split...")

train_idx = int(len(features_df) * 0.70)
val_idx = int(len(features_df) * 0.85)

train_df = features_df.iloc[:train_idx]
val_df = features_df.iloc[train_idx:val_idx]
test_df = features_df.iloc[val_idx:]

print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

X_train, y_train = create_sequences(train_scaled, LOOKBACK, HORIZON, TARGET_COL_INDEX)
X_val, y_val = create_sequences(val_scaled, LOOKBACK, HORIZON, TARGET_COL_INDEX)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, HORIZON, TARGET_COL_INDEX)

if X_train.size == 0 or X_val.size == 0:
    print("ERROR: Not enough data to create sequences with the current LOOKBACK/HORIZON. Reduce LOOKBACK or provide more data.")
    sys.exit(1)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


#################################################################
### 4. NEW: Transformer Helper Layers
#################################################################

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, max_steps=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        P = np.zeros((1, max_steps, embed_dim))
        for pos in range(max_steps):
            for i in range(0, embed_dim, 2):
                P[0, pos, i] = np.sin(pos / (10000.0 ** ((2 * i) / embed_dim)))
                if i + 1 < embed_dim:
                    P[0, pos, i + 1] = np.cos(pos / (10000.0 ** ((2 * (i + 1)) / embed_dim)))
        self.P = tf.constant(P, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.P[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        })
        return config


#################################################################
### 5. NEW: KerasTuner Model Builder
#################################################################

def build_model(hp, model_name):
    inputs = Input(shape=(LOOKBACK, N_FEATURES))

    if model_name in ['lstm', 'gru', 'bilstm']:
        units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
        dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
        rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1)
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
        dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
        rec_dropout_2 = hp.Float('rec_dropout_2', min_value=0.0, max_value=0.5, step=0.1)

        if model_name == 'lstm':
            x = LSTM(units_1, return_sequences=True, recurrent_dropout=rec_dropout_1)(inputs)
            x = Dropout(dropout_1)(x)
            x = LSTM(units_2, recurrent_dropout=rec_dropout_2)(x)
            x = Dropout(dropout_2)(x)
        elif model_name == 'gru':
            x = GRU(units_1, return_sequences=True, recurrent_dropout=rec_dropout_1)(inputs)
            x = Dropout(dropout_1)(x)
            x = GRU(units_2, recurrent_dropout=rec_dropout_2)(x)
            x = Dropout(dropout_2)(x)
        else:  # bilstm
            x = Bidirectional(LSTM(units_1, return_sequences=True, recurrent_dropout=rec_dropout_1))(inputs)
            x = Dropout(dropout_1)(x)
            x = Bidirectional(LSTM(units_2, recurrent_dropout=rec_dropout_2))(x)
            x = Dropout(dropout_2)(x)

    elif model_name == 'cnn_lstm':
        filters = hp.Int('filters', min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice('kernel_size', values=[3, 5])
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='causal')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout)(x)
        x = LSTM(lstm_units)(x)
        x = Dropout(dropout)(x)

    elif model_name == 'transformer':
        embed_dim = hp.Int('embed_dim', min_value=32, max_value=96, step=32)
        num_heads = hp.Choice('num_heads', values=[2, 4])
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        num_blocks = hp.Int('num_blocks', min_value=1, max_value=4, step=1)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)
        x = Dense(embed_dim)(inputs)
        x = PositionalEncoding(embed_dim, max_steps=LOOKBACK)(x)
        for _ in range(num_blocks):
            x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        x = Dense(ff_dim, activation='relu')(x)
        x = Dropout(dropout)(x)
    else:
        raise ValueError("Unknown model name")

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


#################################################################
### 6. NEW: Hyperparameter Tuning Loop (Faster, checkpointed)
#################################################################
print("\nStep 6: Starting Hyperparameter Tuning...")

models_to_run = ['lstm', 'gru', 'bilstm', 'cnn_lstm', 'transformer']
all_best_hps = {}
all_best_models = {}
all_tuning_times = {}

for model_name in models_to_run:
    print(f"\n--- Tuning Model: {model_name.upper()} ---")
    start_time = time.time()

    tuner = RandomSearch(
        lambda hp: build_model(hp, model_name),
        objective='val_loss',
        max_trials=3,            # reduced to speed up tuning
        executions_per_trial=1,
        directory=TUNER_DIR,
        project_name=f"{model_name}_tuning"
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Ensure tuner subdir exists and add checkpoint to save best trial model
    tuner_subdir = os.path.join(TUNER_DIR, f"{model_name}_tuning")
    os.makedirs(tuner_subdir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(tuner_subdir, "best_trial_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    tuner.search(
        X_train, y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint_cb],
        verbose=1
    )

    tuning_time = time.time() - start_time
    all_tuning_times[model_name] = tuning_time / 60.0

    # Load best hyperparams and model from tuner state (robust)
    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)
        best_models = tuner.get_best_models(num_models=1)
        all_best_hps[model_name] = best_hps[0] if best_hps else None
        all_best_models[model_name] = best_models[0] if best_models else None
    except Exception as e:
        print(f"Warning: no successful trials for {model_name}. Skipping. ({e})")
        all_best_hps[model_name] = None
        all_best_models[model_name] = None

    print(f"--- Best HPs for {model_name.upper()} ---")
    if all_best_hps[model_name] is not None:
        print(all_best_hps[model_name].values)
    else:
        print("No best hyperparameters (no successful trials).")
    print(f"Tuning time: {all_tuning_times[model_name]:.2f} minutes")

    # Do NOT remove tuner directory here so checkpoint and trial data remain for resume.


#################################################################
### 7. Final Evaluation on Holdout Test Set
#################################################################
print("\nStep 7: Final Evaluation on Holdout Test Set...")

final_test_results = {}
final_error_summaries = {}

for model_name, model in all_best_models.items():
    if model is None:
        print(f"Skipping evaluation for {model_name} because no model was returned by the tuner.")
        continue

    print(f"\n--- Evaluating Final Model: {model_name.upper()} ---")

    model_save_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    final_model_path = os.path.join(model_save_dir, "FINAL_best_model.keras")
    model.save(final_model_path)

    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE)

    dummy_pred = np.zeros((len(y_pred_scaled), N_FEATURES))
    dummy_test = np.zeros((len(y_test), N_FEATURES))
    dummy_pred[:, TARGET_COL_INDEX] = y_pred_scaled.flatten()
    dummy_test[:, TARGET_COL_INDEX] = y_test.flatten()

    y_pred_unscaled = scaler.inverse_transform(dummy_pred)[:, TARGET_COL_INDEX]
    y_test_unscaled = scaler.inverse_transform(dummy_test)[:, TARGET_COL_INDEX]

    y_pred_unscaled[y_pred_unscaled < 0] = 0

    final_test_results[model_name] = calculate_metrics(y_test_unscaled, y_pred_unscaled)
    errors = y_test_unscaled - y_pred_unscaled
    final_error_summaries[model_name] = pd.Series(errors).describe()

    print(f"--- Results for {model_name.upper()} ---")
    print(pd.DataFrame.from_dict(final_test_results[model_name], orient='index', columns=['Value']))
    print("\n--- Error Statistical Summary ---")
    print(final_error_summaries[model_name])

    plot_actual_vs_predicted(y_test_unscaled, y_pred_unscaled, model_name, model_save_dir)
    plot_error_distribution(y_test_unscaled, y_pred_unscaled, model_name, model_save_dir)


#################################################################
### 8. SHAP (left disabled for speed)
#################################################################
# (SHAP code intentionally disabled)


#################################################################
### 9. Final Report
#################################################################
print("\n--- Project Complete ---")

final_results_df = pd.DataFrame.from_dict(final_test_results, orient='index')
final_results_df.to_csv(os.path.join(RESULTS_DIR, "final_test_set_results.csv"))

tuning_times_df = pd.DataFrame.from_dict(all_tuning_times, orient='index', columns=['Tuning Time (min)'])
tuning_times_df.to_csv(os.path.join(RESULTS_DIR, "model_tuning_times.csv"))

print("\n\n--- Final Model Results (on Holdout Test Set) ---")
try:
    from tabulate import tabulate
    final_results_df = final_results_df.sort_values(by='RMSE (MW)')
    print(tabulate(final_results_df, headers='keys', tablefmt='pipe', floatfmt=".3f"))
except ImportError:
    print(final_results_df.sort_values(by='RMSE (MW)'))

print("\n\n--- Model Tuning Times ---")
try:
    tuning_times_df = tuning_times_df.sort_values(by='Tuning Time (min)')
    print(tabulate(tuning_times_df, headers='keys', tablefmt='pipe', floatfmt=".2f"))
except ImportError:
    print(tuning_times_df.sort_values(by='Tuning Time (min)'))

plot_model_comparison(final_results_df, 'RMSE (MW)', RESULTS_DIR)
plot_model_comparison(final_results_df, 'MAE (MW)', RESULTS_DIR)
plot_model_comparison(final_results_df, 'R²', RESULTS_DIR)

print(f"\nAll plots, models, and CSV results saved to: '{RESULTS_DIR}'")
# ...existing code...
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Configuration
LOOKBACK = 72
BATCH_SIZE = 64
DATA_FILE = 'time_series_60min_singleindex.csv'

# Global variables
solar_model = None
wind_model = None
df_data = None


# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def create_sequences(data, lookback):
    """Create sequences for model prediction"""
    if len(data) < lookback:
        padding = np.zeros((lookback - len(data), data.shape[1]))
        data = np.vstack([padding, data])
    return data[-lookback:].reshape(1, lookback, -1)


def load_models():
    """Load models and dataset"""
    global solar_model, wind_model, df_data

    try:
        solar_model_path = 'FINAL_best_model_solar.keras'
        wind_model_path = 'FINAL_best_model_wind.keras'

        # Load solar model
        if os.path.exists(solar_model_path):
            solar_model = tf.keras.models.load_model(solar_model_path)
            print(f"✓ Loaded solar model: {solar_model_path}")
        else:
            print(f"⚠ Solar model not found: {solar_model_path}")

        # Load wind model
        if os.path.exists(wind_model_path):
            wind_model = tf.keras.models.load_model(wind_model_path)
            print(f"✓ Loaded wind model: {wind_model_path}")
        else:
            print(f"⚠ Wind model not found: {wind_model_path}")

        # Load dataset
        if os.path.exists(DATA_FILE):
            print(f"📂 Found dataset: {DATA_FILE}")
            df_data = pd.read_csv(DATA_FILE, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
            df_data.index = df_data.index + pd.Timedelta(hours=5, minutes=30)
            print(f"✓ Loaded {len(df_data)} records (timestamps converted to IST)")
        else:
            print(f"⚠ Data file not found: {DATA_FILE}")

    except Exception as e:
        print(f"Error loading models: {e}")


def prepare_features(df, target_col):
    """Prepare dataset features"""
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    if target_col in df.columns:
        df['power'] = df[target_col].clip(lower=0)
    else:
        df['power'] = 0

    df['load'] = df.get('DE_load_actual_entsoe_transparency', 0)

    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    return df[['power', 'load', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'solar_model_loaded': bool(solar_model is not None),
        'wind_model_loaded': bool(wind_model is not None),
        'data_loaded': bool(df_data is not None)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict next-hour solar and wind output"""
    try:
        data = request.json
        demand = float(data.get('demand', 0))

        # Fallback if models/data missing
        if solar_model is None or wind_model is None or df_data is None:
            print("⚠ Models or data missing — returning mock predictions")
            solar_pred = float(np.random.uniform(1000, 5000))
            wind_pred = float(np.random.uniform(2000, 6000))
            total_available = float(solar_pred + wind_pred)
            shortage = float(max(0, demand - total_available))
            
            result = {
                'solar_pred': solar_pred,
                'wind_pred': wind_pred,
                'total_available': total_available,
                'demand': demand,
                'shortage': shortage,
                'can_meet_demand': bool(total_available >= demand),
                'timestamp': datetime.now().isoformat()
            }
            print("🔍 Returning mock prediction:", result)
            return jsonify(result)

        # Actual model prediction
        recent_data = df_data.tail(LOOKBACK + 10)

        # Solar prediction
        solar_pred = 0.0
        if solar_model is not None:
            try:
                solar_features = prepare_features(recent_data, 'DE_solar_generation_actual')
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(solar_features)
                X = create_sequences(scaled, LOOKBACK)
                scaled_pred = solar_model.predict(X, batch_size=1, verbose=0)
                dummy = np.zeros((1, 6))
                dummy[0, 0] = scaled_pred[0, 0]
                solar_pred = float(scaler.inverse_transform(dummy)[0, 0])
                solar_pred = float(max(0, solar_pred))
            except Exception as e:
                print(f"Solar prediction error: {e}")
                solar_pred = 0.0

        # Wind prediction
        wind_pred = 0.0
        if wind_model is not None:
            try:
                wind_features = prepare_features(recent_data, 'DE_wind_generation_actual')
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(wind_features)
                X = create_sequences(scaled, LOOKBACK)
                scaled_pred = wind_model.predict(X, batch_size=1, verbose=0)
                dummy = np.zeros((1, 6))
                dummy[0, 0] = scaled_pred[0, 0]
                wind_pred = float(scaler.inverse_transform(dummy)[0, 0])
                wind_pred = float(max(0, wind_pred))
            except Exception as e:
                print(f"Wind prediction error: {e}")
                wind_pred = 0.0

        total_available = float(solar_pred + wind_pred)
        shortage = float(max(0, demand - total_available))

        result = {
            'solar_pred': solar_pred,
            'wind_pred': wind_pred,
            'total_available': total_available,
            'demand': demand,
            'shortage': shortage,
            'can_meet_demand': bool(total_available >= demand),
            'timestamp': datetime.now().isoformat()
        }
        
        print("🔍 Returning prediction:", result)
        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical', methods=['GET'])
def get_historical():
    """Return last 72h + 1-hour forecast"""
    try:
        if df_data is None:
            return jsonify({'error': 'Data not loaded'}), 500

        hours = int(request.args.get('hours', 72))
        recent = df_data.tail(hours)

        solar_data, wind_data = [], []

        # Historical data
        if 'DE_solar_generation_actual' in recent.columns:
            for idx, val in recent['DE_solar_generation_actual'].fillna(0).items():
                solar_data.append({
                    'timestamp': idx.isoformat(),
                    'value': float(val)
                })

        if 'DE_wind_generation_actual' in recent.columns:
            for idx, val in recent['DE_wind_generation_actual'].fillna(0).items():
                wind_data.append({
                    'timestamp': idx.isoformat(),
                    'value': float(val)
                })

        # Forecast point
        if solar_model is not None and wind_model is not None:
            recent_data = df_data.tail(LOOKBACK + 10)

            # Solar forecast
            try:
                solar_features = prepare_features(recent_data, 'DE_solar_generation_actual')
                scaler_solar = MinMaxScaler()
                solar_scaled = scaler_solar.fit_transform(solar_features)
                X_solar = create_sequences(solar_scaled, LOOKBACK)
                solar_pred_scaled = solar_model.predict(X_solar, batch_size=1, verbose=0)
                dummy_solar = np.zeros((1, 6))
                dummy_solar[0, 0] = solar_pred_scaled[0, 0]
                solar_pred = float(scaler_solar.inverse_transform(dummy_solar)[0, 0])
                solar_pred = float(max(0, solar_pred))
            except Exception as e:
                print(f"Solar forecast error: {e}")
                solar_pred = 0.0

            # Wind forecast
            try:
                wind_features = prepare_features(recent_data, 'DE_wind_generation_actual')
                scaler_wind = MinMaxScaler()
                wind_scaled = scaler_wind.fit_transform(wind_features)
                X_wind = create_sequences(wind_scaled, LOOKBACK)
                wind_pred_scaled = wind_model.predict(X_wind, batch_size=1, verbose=0)
                dummy_wind = np.zeros((1, 6))
                dummy_wind[0, 0] = wind_pred_scaled[0, 0]
                wind_pred = float(scaler_wind.inverse_transform(dummy_wind)[0, 0])
                wind_pred = float(max(0, wind_pred))
            except Exception as e:
                print(f"Wind forecast error: {e}")
                wind_pred = 0.0

            # Use current IST time + 1 hour
            forecast_time = datetime.now() + timedelta(hours=1)
            print("📊 Forecast (IST):", forecast_time)
            print("☀️ Solar Forecast:", solar_pred)
            print("💨 Wind Forecast:", wind_pred)

            # Append forecast data
            solar_data.append({
                'timestamp': forecast_time.isoformat(),
                'value': solar_pred,
                'type': 'forecast'
            })
            wind_data.append({
                'timestamp': forecast_time.isoformat(),
                'value': wind_pred,
                'type': 'forecast'
            })

        return jsonify({'solar': solar_data, 'wind': wind_data})

    except Exception as e:
        print(f"❌ Historical data error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------
# Run Server
# -----------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("🌞 Renewable Energy Forecasting Backend")
    print("=" * 60)
    load_models()
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

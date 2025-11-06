"""
Demo Backend with Mock Predictions
Use this for testing the frontend without trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

def generate_mock_historical_data(hours=73):
    """Generate realistic mock historical data"""
    solar_data = []
    wind_data = []
    
    start_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        hour = timestamp.hour
        
        # Solar: Peak at noon, zero at night
        solar_value = max(0, 5000 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 200))
        
        # Wind: More variable, higher at night
        wind_value = 3000 + 1000 * np.sin(hour * np.pi / 12) + np.random.normal(0, 400)
        wind_value = max(0, wind_value)
        
        solar_data.append({
            'timestamp': timestamp.isoformat(),
            'value': float(solar_value)
        })
        
        wind_data.append({
            'timestamp': timestamp.isoformat(),
            'value': float(wind_value)
        })
    
    return solar_data, wind_data

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mode': 'demo',
        'solar_model_loaded': False,
        'wind_model_loaded': False,
        'data_loaded': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Mock prediction endpoint"""
    try:
        data = request.json
        demand = data.get('demand', 0)
        
        # Generate realistic predictions
        hour = datetime.now().hour
        
        # Solar: Peak during day
        solar_pred = max(0, 4500 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 150))
        
        # Wind: More consistent, slightly higher at night
        wind_pred = 3200 + 800 * np.sin(hour * np.pi / 12) + np.random.normal(0, 300)
        wind_pred = max(0, wind_pred)
        
        # Calculate totals and distribution
        total_available = solar_pred + wind_pred
        
        if demand <= total_available:
            if total_available > 0:
                solar_share = (solar_pred / total_available) * demand
                wind_share = (wind_pred / total_available) * demand
            else:
                solar_share = 0
                wind_share = 0
            shortage = 0
            can_meet_demand = True
        else:
            solar_share = solar_pred
            wind_share = wind_pred
            shortage = demand - total_available
            can_meet_demand = False
        
        return jsonify({
            'solar_pred': float(solar_pred),
            'wind_pred': float(wind_pred),
            'total_available': float(total_available),
            'demand': float(demand),
            'solar_share': float(solar_share),
            'wind_share': float(wind_share),
            'shortage': float(shortage),
            'can_meet_demand': can_meet_demand,
            'timestamp': datetime.now().isoformat(),
            'mode': 'demo'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical', methods=['GET'])
def get_historical():
    """Get mock historical data for charts"""
    try:
        hours = int(request.args.get('hours', 73))
        solar_data, wind_data = generate_mock_historical_data(hours)
        
        return jsonify({
            'solar': solar_data,
            'wind': wind_data,
            'mode': 'demo'
        })
        
    except Exception as e:
        print(f"Historical data error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("Renewable Energy Forecasting Backend - DEMO MODE")
    print("="*60)
    print(" Using mock predictions - no ML models loaded")
    print("This is for testing the frontend interface only")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
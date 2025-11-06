# Deployment Guide

## Quick Start Guide

### Option 1: Demo Mode (No Models Required)

Test the frontend interface with mock data:

```bash
# Backend (Demo)
cd backend
pip install Flask flask-cors numpy
python app_demo.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` and test with demand values like 5000 MW.

### Option 2: Full Production Setup

#### Step 1: Train Models

```bash
# Download dataset
wget https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv

# Train solar models
python train_solar_models_v3_fast.py

# Train wind models
python train_wind_models.py
```

This will create:
- `project_results_v3_fast/lstm/FINAL_best_model.keras`
- `project_results_wind_v1/lstm/FINAL_best_model.keras`

#### Step 2: Setup Backend

```bash
cd backend
./setup.sh
source venv/bin/activate
python app.py
```

#### Step 3: Setup Frontend

```bash
cd frontend
./setup.sh
npm run dev
```

## Production Deployment

### Backend (Flask) Deployment

#### Using Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Docker

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t renewable-energy-backend .
docker run -p 5000:5000 renewable-energy-backend
```

### Frontend (React) Deployment

#### Build for Production

```bash
npm run build
```

This creates an optimized build in `dist/` directory.

#### Deploy to Netlify

1. Push code to GitHub
2. Connect repository to Netlify
3. Set build command: `npm run build`
4. Set publish directory: `dist`
5. Add environment variable: `VITE_API_URL=https://your-backend-url.com`

#### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

Follow prompts to deploy.

#### Serve with Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Environment Configuration

### Backend Environment Variables

Create `backend/.env`:

```env
FLASK_ENV=production
FLASK_DEBUG=False
DATA_FILE=time_series_60min_singleindex.csv
SOLAR_MODEL_PATH=project_results_v3_fast/lstm/FINAL_best_model.keras
WIND_MODEL_PATH=project_results_wind_v1/lstm/FINAL_best_model.keras
```

### Frontend Environment Variables

Create `frontend/.env.production`:

```env
VITE_API_URL=https://your-backend-api.com/api
```

## Performance Optimization

### Backend Optimizations

1. **Model Caching**: Models are loaded once at startup
2. **Batch Predictions**: Use batch size of 1 for single predictions
3. **Data Preprocessing**: Cache scaler objects
4. **Gzip Compression**: Enable in production

```python
from flask_compress import Compress
Compress(app)
```

### Frontend Optimizations

1. **Code Splitting**: Vite handles automatically
2. **Lazy Loading**: Load charts only when needed
3. **Debouncing**: Add input debouncing for API calls
4. **Caching**: Implement SWR or React Query

```javascript
// Example: Debounced prediction
const debouncedPredict = debounce(handlePredict, 500);
```

## Monitoring & Logging

### Backend Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Prediction made: %.2f MW", prediction)
```

### Frontend Error Tracking

Integrate Sentry:

```bash
npm install @sentry/react
```

```javascript
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: "your-sentry-dsn",
  environment: "production"
});
```

## Scaling Strategies

### Horizontal Scaling

1. **Load Balancer**: Use Nginx or HAProxy
2. **Multiple Backend Instances**: Run multiple Gunicorn workers
3. **Database**: Store predictions in Redis for caching

### Vertical Scaling

1. **GPU Support**: Use TensorFlow GPU for faster inference
2. **Model Optimization**: Use TensorFlow Lite
3. **Memory Management**: Implement connection pooling

## Security Best Practices

### Backend Security

```python
# Rate limiting
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])

@app.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    pass
```

### CORS Configuration

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-domain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

### HTTPS

Always use HTTPS in production:
- Use Let's Encrypt for free SSL certificates
- Configure Nginx SSL termination
- Redirect HTTP to HTTPS

## Health Checks

### Backend Health Endpoint

```python
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': solar_model is not None and wind_model is not None
    })
```

### Frontend Health Check

Add to CI/CD pipeline:

```bash
curl http://localhost:3000 | grep "Renewable Energy"
```

## Backup & Recovery

1. **Model Backups**: Regular backups of trained models
2. **Data Backups**: Backup historical data
3. **Configuration Backups**: Version control all configs

## Troubleshooting

### Common Issues

**Backend won't start**:
```bash
# Check port availability
lsof -i :5000

# Check logs
tail -f logs/app.log
```

**Frontend can't connect to backend**:
- Verify CORS configuration
- Check API URL in environment variables
- Ensure backend is running

**Predictions are inaccurate**:
- Retrain models with recent data
- Check data preprocessing
- Verify scaler parameters

## Cost Optimization

### Cloud Deployment Costs

- **Backend**: ~$20-50/month (1-2 GB RAM)
- **Frontend**: Free (Netlify/Vercel free tier)
- **Total**: ~$20-50/month

### Resource Requirements

- **CPU**: 2 cores minimum
- **RAM**: 2 GB minimum (4 GB recommended)
- **Storage**: 10 GB for models and data
- **Bandwidth**: ~1 TB/month for 10k users

## Maintenance Schedule

- **Daily**: Monitor logs and error rates
- **Weekly**: Check prediction accuracy
- **Monthly**: Retrain models with new data
- **Quarterly**: Update dependencies and security patches

---

For questions or issues, refer to the main README.md or open an issue.
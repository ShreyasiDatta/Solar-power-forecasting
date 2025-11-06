# Quick Reference Guide

## 🚀 Getting Started in 5 Minutes

### Instant Demo (No Models Required)

```bash
# 1. Start Backend Demo
cd backend
pip install Flask flask-cors numpy
python app_demo.py
# Server runs on http://localhost:5000

# 2. Start Frontend (new terminal)
cd frontend
npm install
npm run dev
# Opens http://localhost:3000 automatically

# 3. Test the Dashboard
# Enter: 5000 (in the demand input)
# Click: "Calculate Energy Balance"
# View: Predictions, charts, and distribution
```

## 📋 Common Commands

### Backend
```bash
# Setup
cd backend
./setup.sh

# Run Production
source venv/bin/activate
python app.py

# Run Demo
python app_demo.py

# Install Dependencies
pip install -r requirements.txt
```

### Frontend
```bash
# Setup
cd frontend
./setup.sh

# Development
npm run dev

# Production Build
npm run build

# Preview Build
npm run preview
```

## 🔧 Configuration Quick Reference

### Backend Config
File: `backend/.env`
```env
DATA_FILE=time_series_60min_singleindex.csv
SOLAR_MODEL_PATH=project_results_v3_fast/lstm/FINAL_best_model.keras
WIND_MODEL_PATH=project_results_wind_v1/lstm/FINAL_best_model.keras
```

### Frontend Config
File: `frontend/.env`
```env
VITE_API_URL=http://localhost:5000/api
```

## 🎯 Test Values

Try these demand values:
- **3000 MW** - Usually met by renewables
- **5000 MW** - Balanced scenario
- **8000 MW** - May show shortage
- **10000 MW** - Shortage likely

## 📊 Expected Results

### Example Prediction
```
Demand: 5000 MW
Solar: 2500 MW (50%)
Wind: 3200 MW (64%)
Total: 5700 MW
Result: ✓ Can meet demand
Solar Share: 2193 MW
Wind Share: 2807 MW
```

## 🐛 Quick Troubleshooting

### Backend Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Try different port
export PORT=5001
python app.py
```

### Frontend Can't Connect
1. Check backend is running: `curl http://localhost:5000/api/health`
2. Update API URL in frontend `.env`
3. Check browser console for CORS errors

### Models Not Loading
- Ensure model files exist in correct paths
- Use demo mode: `python app_demo.py`
- Check file permissions

## 📦 File Locations

```
backend/
├── app.py              ← Production backend
├── app_demo.py         ← Demo backend
├── requirements.txt    ← Python deps

frontend/
├── src/App.jsx         ← Main component
├── src/App.css         ← Styles
├── package.json        ← Node deps

Models: (if trained)
├── project_results_v3_fast/lstm/FINAL_best_model.keras
├── project_results_wind_v1/lstm/FINAL_best_model.keras

Data: (if available)
└── time_series_60min_singleindex.csv
```

## 🌐 API Endpoints

```
GET  /api/health
→ Status check

POST /api/predict
Body: {"demand": 5000}
→ Solar, wind predictions + balance

GET  /api/historical?hours=72
→ Last 72 hours of data
```

## 🎨 Color Reference

```css
Solar:  #FDB813  /* Yellow */
Wind:   #00A9CE  /* Blue */
Total:  #A855F7  /* Purple */
Good:   #10B981  /* Green */
Alert:  #F59E0B  /* Orange */
Error:  #EF4444  /* Red */
```

## ⌨️ Keyboard Shortcuts

Frontend (in browser):
- `Ctrl + R` - Refresh data
- `F5` - Reload page
- `F12` - Open DevTools

## 📱 Mobile Testing

```bash
# Get local IP
ipconfig getifaddr en0  # Mac
ip addr show           # Linux

# Update frontend .env
VITE_API_URL=http://YOUR_IP:5000/api

# Access from mobile
http://YOUR_IP:3000
```

## 🔒 Security Checklist

- [ ] Change default SECRET_KEY in production
- [ ] Enable HTTPS
- [ ] Set proper CORS origins
- [ ] Add rate limiting
- [ ] Use environment variables
- [ ] Update dependencies regularly

## 📈 Performance Tips

1. **Backend**: Use Gunicorn with 4 workers
2. **Frontend**: Run `npm run build` for production
3. **Caching**: Implement Redis for predictions
4. **CDN**: Use CDN for static assets

## 🎓 Learning Resources

- [React Docs](https://react.dev)
- [Flask Docs](https://flask.palletsprojects.com)
- [TensorFlow Docs](https://tensorflow.org)
- [Recharts Examples](https://recharts.org)

## 💡 Pro Tips

1. Use demo mode to test frontend without models
2. Check `/api/health` before troubleshooting
3. Monitor browser console for errors
4. Use React DevTools for debugging
5. Test with different demand values
6. Check network tab for API responses

## 📞 Support

1. Check README.md for detailed docs
2. Review DEPLOYMENT.md for production
3. See PROJECT_SUMMARY.md for overview
4. Open browser DevTools for errors
5. Check backend logs

## ✅ Quick Test Checklist

- [ ] Backend starts without errors
- [ ] Frontend loads at localhost:3000
- [ ] Health endpoint returns 200
- [ ] Can enter demand value
- [ ] Prediction button works
- [ ] Charts display correctly
- [ ] Results show properly
- [ ] No console errors

## 🚦 Status Indicators

**Backend Running**:
```bash
curl http://localhost:5000/api/health
# Should return: {"status": "healthy"}
```

**Frontend Running**:
```bash
curl http://localhost:3000
# Should return HTML
```

---

**Quick Start**: Run demo mode → Test with 5000 MW → Explore dashboard!
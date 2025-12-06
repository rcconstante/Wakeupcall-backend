# WakeUpCall Backend

Sleep apnea risk assessment API backend for the WakeUpCall Android app.

## Deployment on Railway

### Quick Deploy

1. Push your code to a GitHub repository
2. Connect the repository to Railway
3. Railway will automatically detect the Python app and deploy it

### Environment Variables

Set these environment variables in Railway:

- `SECRET_KEY`: A secure random string for Flask sessions (optional - auto-generated if not set)
- `DATABASE_PATH`: Path to SQLite database (optional - defaults to `wakeup_call.db`)
- `FLASK_DEBUG`: Set to `false` for production

### Files for Railway Deployment

- `Procfile`: Defines the startup command using Gunicorn
- `railway.json`: Railway-specific configuration
- `requirements.txt`: Python dependencies

### Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### API Endpoints

- `GET /`: Health check
- `POST /auth/signup`: User registration
- `POST /auth/login`: User login
- `POST /auth/logout`: User logout
- `GET /auth/verify`: Verify auth token
- `POST /survey/submit`: Submit survey data
- `GET /survey/get-latest`: Get latest survey results
- `POST /survey/generate-pdf`: Generate PDF report

### Model Files Required

Make sure these files are in the `backend/` directory:
- `lightgbm_sleep_apnea_model.pkl`: The trained LightGBM model
- `scaler.pkl`: (Optional) Feature scaler if required by the model

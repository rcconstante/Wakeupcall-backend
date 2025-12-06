from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sqlite3
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from recommendation_engine import RecommendationEngine

# Try joblib first (more compatible with scikit-learn models), fall back to pickle
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    USE_JOBLIB = False
    print("⚠️ joblib not installed, using pickle (may have compatibility issues with newer Python versions)")

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Enable CORS for Android app to access the API
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Database setup - use environment variable for Railway persistence
DATABASE = os.environ.get('DATABASE_PATH', 'wakeup_call.db')

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with users table"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Auth tokens table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auth_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User survey data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_surveys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            -- Demographics
            age INTEGER,
            sex TEXT,
            height_cm REAL,
            weight_kg REAL,
            neck_circumference_cm REAL,
            bmi REAL,
            -- Medical history
            hypertension INTEGER,
            diabetes INTEGER,
            depression INTEGER DEFAULT 0,
            smokes INTEGER,
            alcohol INTEGER,
            -- Survey scores
            ess_score INTEGER,
            berlin_score INTEGER,
            stopbang_score INTEGER,
            -- ML prediction
            osa_probability REAL,
            risk_level TEXT,
            -- Google Fit data
            daily_steps INTEGER,
            average_daily_steps INTEGER,
            sleep_duration_hours REAL,
            weekly_steps_json TEXT,
            weekly_sleep_json TEXT,
            -- Additional survey fields
            snoring_level TEXT,
            snoring_frequency TEXT,
            snoring_bothers_others INTEGER,
            tired_during_day TEXT,
            tired_after_sleep TEXT,
            feels_sleepy_daytime INTEGER,
            nodded_off_driving INTEGER,
            physical_activity_time TEXT,
            -- ESS individual scores
            ess_sitting_reading INTEGER,
            ess_watching_tv INTEGER,
            ess_public_sitting INTEGER,
            ess_passenger_car INTEGER,
            ess_lying_down_afternoon INTEGER,
            ess_talking INTEGER,
            ess_after_lunch INTEGER,
            ess_traffic_stop INTEGER,
            -- STOP-BANG individual responses
            stopbang_snoring INTEGER,
            stopbang_tired INTEGER,
            stopbang_observed_apnea INTEGER,
            stopbang_pressure INTEGER,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Add missing columns if table already exists (migration for existing deployments)
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN snoring_level TEXT')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN snoring_frequency TEXT')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN snoring_bothers_others INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN tired_during_day TEXT')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN tired_after_sleep TEXT')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN feels_sleepy_daytime INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN nodded_off_driving INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN physical_activity_time TEXT')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_sitting_reading INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_watching_tv INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_public_sitting INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_passenger_car INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_lying_down_afternoon INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_talking INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_after_lunch INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN ess_traffic_stop INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN stopbang_snoring INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN stopbang_tired INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN stopbang_observed_apnea INTEGER')
    except: pass
    try:
        cursor.execute('ALTER TABLE user_surveys ADD COLUMN stopbang_pressure INTEGER')
    except: pass
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

# Initialize database on startup
init_db()

def require_auth(f):
    """Decorator to require authentication token (supports guest mode)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No authorization token provided', 'success': False}), 401
        
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        # Handle guest mode tokens
        if token.startswith('guest_token_'):
            # Guest user - create temporary user context
            request.current_user = {
                'id': -1,  # Guest user ID
                'email': 'guest@wakeupcall.app',
                'first_name': 'Guest',
                'last_name': 'User',
                'is_guest': True
            }
            return f(*args, **kwargs)
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.id, u.email, u.first_name, u.last_name 
            FROM users u 
            JOIN auth_tokens t ON u.id = t.user_id 
            WHERE t.token = ? AND t.expires_at > ?
        ''', (token, datetime.now()))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid or expired token', 'success': False}), 401
        
        # Pass user info to the route
        request.current_user = {
            'id': user[0],
            'email': user[1],
            'first_name': user[2],
            'last_name': user[3],
            'is_guest': False
        }
        
        return f(*args, **kwargs)
    
    return decorated_function

# Load the trained model pipeline
# Using final_lgbm_pipeline.pkl which contains the complete pipeline
model = None
scaler = None

# Model paths - prioritize lightgbm_sleep_apnea_model.pkl
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), 'lightgbm_sleep_apnea_model.pkl')
]

SCALER_PATHS = [
    os.path.join(os.path.dirname(__file__), 'scaler.pkl'),  # backend folder
    os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl'),  # model folder
]

# Load model
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            if USE_JOBLIB:
                loaded_data = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
            
            # Check if it's a dict with model and scaler
            if isinstance(loaded_data, dict):
                model = loaded_data.get('model')
                if not scaler:  # Only use dict scaler if separate file not found
                    scaler = loaded_data.get('scaler')
                print(f"✅ Model loaded from dictionary: {model_path}")
            else:
                # It's just the model object (pipeline)
                model = loaded_data
                print(f"✅ Model loaded from: {model_path}")
            
            if model is not None:
                break
        except Exception as e:
            print(f"⚠️ Error loading model from {model_path}: {e}")
            # Try with pickle and latin1 encoding for compatibility
            if 'WakeUpCall_3Class5Fold' in model_path:
                try:
                    print(f"   Attempting compatibility mode for {os.path.basename(model_path)}...")
                    with open(model_path, 'rb') as f:
                        loaded_data = pickle.load(f, encoding='latin1')
                    model = loaded_data
                    print(f"✅ Model loaded with compatibility mode: {model_path}")
                    break
                except Exception as e2:
                    print(f"   Compatibility mode also failed: {e2}")

# Load scaler separately (optional, for backwards compatibility with old models)
for scaler_path in SCALER_PATHS:
    if os.path.exists(scaler_path):
        try:
            if USE_JOBLIB:
                scaler = joblib.load(scaler_path)
            else:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            print(f"✅ Scaler loaded from: {scaler_path}")
            break
        except Exception as e:
            print(f"⚠️ Error loading scaler from {scaler_path}: {e}")

if model is None:
    print("⚠️ Model file not found. Expected one of:")
    for p in MODEL_PATHS:
        print(f"   - {p}")
    print("Please ensure lgbm_model.pkl exists in backend or model folder!")
else:
    print("✅ Model loaded successfully!")

if scaler is None:
    print("⚠️ Scaler file not found. Expected one of:")
    for p in SCALER_PATHS:
        print(f"   - {p}")
    print("Please ensure scaler.pkl exists in backend or model folder!")
else:
    print("✅ Scaler loaded successfully!")

def generate_ml_recommendation(osa_probability, risk_level, age, bmi, neck_cm, hypertension, diabetes, smokes, alcohol, ess_score, berlin_score, stopbang_score, sleep_duration=7.0, daily_steps=5000, physical_activity_time=None):
    """Generate personalized recommendations using comprehensive recommendation engine.
    Limits recommendations based on risk level to avoid overwhelming users."""
    
    # Use the new RecommendationEngine
    sex = 1  # Default to male (conservative for OSA risk)
    recommendations = RecommendationEngine.generate_recommendations(
        age=age,
        sex=sex,
        bmi=bmi,
        neck_cm=neck_cm,
        hypertension=hypertension,
        diabetes=diabetes,
        smokes=smokes,
        alcohol=alcohol,
        ess_score=ess_score,
        berlin_score=berlin_score,
        stopbang_score=stopbang_score,
        sleep_duration=sleep_duration,
        daily_steps=daily_steps,
        risk_level=risk_level,
        physical_activity_time=physical_activity_time
    )
    
    # Sort by priority (descending) and limit based on risk level
    recommendations.sort(key=lambda r: r.priority, reverse=True)
    
    # Limit recommendations to avoid overwhelming users
    if "High" in risk_level:
        max_recommendations = 8  # High risk: show top 8 most critical
    elif "Intermediate" in risk_level:
        max_recommendations = 5  # Intermediate risk: show top 5
    else:
        max_recommendations = 4  # Low risk: show top 4
    
    limited_recommendations = recommendations[:max_recommendations]
    
    # Format for API response (pipe-separated)
    return RecommendationEngine.format_for_api(limited_recommendations)

def calculate_top_risk_factors(input_features, osa_probability):
    """Calculate top risk factors based on actual survey data and thresholds"""
    
    risk_factors = []
    
    # Analyze each feature and its contribution to OSA risk
    age = input_features.get('Age', 0)
    sex = input_features.get('Sex', 0)  # 1=male, 0=female
    bmi = input_features.get('BMI', 0)
    neck = input_features.get('Neck_Circumference', 0)
    ess = input_features.get('Epworth_Score', 0)
    berlin = input_features.get('Berlin_Score', 0)
    stopbang = input_features.get('STOPBANG_Total', 0)
    
    # BMI analysis
    if bmi >= 35:
        risk_factors.append(("Severe Obesity", f"BMI {bmi:.1f}", 0.25))
    elif bmi >= 30:
        risk_factors.append(("Obesity", f"BMI {bmi:.1f}", 0.15))
    elif bmi >= 25:
        risk_factors.append(("Overweight", f"BMI {bmi:.1f}", 0.08))
    
    # Neck circumference
    if neck >= 43:
        risk_factors.append(("Large Neck", f"{neck}cm", 0.20))
    elif neck >= 40:
        risk_factors.append(("Thick Neck", f"{neck}cm", 0.12))
    
    # Age factor
    if age >= 65:
        risk_factors.append(("Advanced Age", f"{age} years", 0.15))
    elif age >= 50:
        risk_factors.append(("Middle Age", f"{age} years", 0.08))
    
    # Gender (male higher risk)
    if sex == 1:
        risk_factors.append(("Male Gender", "Higher OSA risk", 0.10))
    
    # Sleepiness
    if ess >= 16:
        risk_factors.append(("Severe Sleepiness", f"ESS {ess}", 0.18))
    elif ess >= 11:
        risk_factors.append(("Moderate Sleepiness", f"ESS {ess}", 0.10))
    
    # Snoring
    if berlin >= 1:
        risk_factors.append(("Snoring Issues", f"Berlin Score {berlin}", 0.12))
    
    # STOP-BANG
    if stopbang >= 6:
        risk_factors.append(("High STOP-BANG", f"Score {stopbang}/8", 0.20))
    elif stopbang >= 4:
        risk_factors.append(("Moderate STOP-BANG", f"Score {stopbang}/8", 0.12))
    
    # Medical conditions
    if input_features.get('Hypertension', 0):
        risk_factors.append(("Hypertension", "High blood pressure", 0.08))
    
    if input_features.get('Diabetes', 0):
        risk_factors.append(("Diabetes", "Blood sugar disorder", 0.06))
    
    if input_features.get('Alcohol', 0):
        risk_factors.append(("Alcohol Use", "Relaxes throat muscles", 0.05))
    
    # Sort by impact and return top factors
    risk_factors.sort(key=lambda x: x[2], reverse=True)
    
    # Format for frontend
    formatted_factors = []
    for i, (name, detail, impact) in enumerate(risk_factors[:6]):  # Top 6 factors
        formatted_factors.append({
            "factor": name,
            "detail": detail,
            "impact": f"{impact*100:.0f}%",
            "priority": "High" if impact >= 0.15 else "Medium" if impact >= 0.08 else "Low"
        })
    
    return formatted_factors

# Feature list (must match lightgbm_sleep_apnea_model.pkl training order)
# Based on model trained with 33 engineered features in exact order
FEATURES = [
    # Demographics
    'Age', 'Sex', 'Age_Group',
    # Anthropometrics
    'Height', 'Weight', 'BMI', 'BMI_Category',
    'Neck_Circumference', 'Neck_Above_40',
    'Neck_Height_Ratio', 'Weight_Height_Ratio',
    # Lifestyle
    'Smokes', 'Alcohol',
    # Sleep symptoms
    'Snoring', 'Sleepiness', 'Epworth_Score', 'ESS_Category',
    # Medical history
    'Hypertension', 'Diabetes', 'Depression',
    # STOP components
    'STOP_Snore', 'STOP_Tired', 'STOP_ObsApnea', 'STOP_Pressure', 'STOP_Count',
    # BANG components
    'BANG_Age', 'BANG_BMI', 'BANG_Neck', 'BANG_Gender', 'BANG_Total',
    # Composite scores
    'STOPBANG', 'Berlin_Score',
    # Composite feature
    'Airway_Composite'
]

# Columns to scale (numerical features) - must match training scaler
NUM_COLS = [
    'Age', 'Height', 'Weight', 'BMI', 'BMI_Category',
    'Neck_Circumference', 'Neck_Height_Ratio', 'Weight_Height_Ratio',
    'Epworth_Score', 'STOP_Count', 'BANG_Total', 'STOPBANG',
    'Berlin_Score', 'Airway_Composite'
]

# ESS Category encoding (must match training label encoder)
ESS_CATEGORY_MAP = {
    'Mild Sleepiness': 0,
    'Moderate Sleepiness': 1,
    'Normal': 2,
    'Severe Sleepiness': 3
}


def calculate_ess_category(ess_score):
    """Calculate ESS category from score (matches training notebook)"""
    if ess_score < 10:
        return "Normal"
    elif ess_score <= 12:
        return "Mild Sleepiness"
    elif ess_score <= 15:
        return "Moderate Sleepiness"
    else:
        return "Severe Sleepiness"


def calculate_bmi_category(bmi):
    """Calculate BMI category (0-3) matching training notebook"""
    if bmi <= 25:
        return 0
    elif bmi <= 30:
        return 1
    elif bmi <= 35:
        return 2
    else:
        return 3


def engineer_features(raw_features):
    """
    Calculate all engineered features from raw input.
    Must match the feature engineering in the training notebook exactly.
    """
    # Extract raw values
    height = raw_features['Height']
    weight = raw_features['Weight']
    bmi = raw_features['BMI']
    neck = raw_features['Neck_Circumference']
    ess_score = raw_features['Epworth_Score']
    alcohol = raw_features['Alcohol']
    snoring = raw_features['Snoring']
    
    # STOP components
    stop_snore = raw_features['STOP_Snore']
    stop_tired = raw_features['STOP_Tired']
    stop_obs = raw_features['STOP_ObsApnea']
    stop_pressure = raw_features['STOP_Pressure']
    
    # BANG components
    bang_age = raw_features['BANG_Age']
    bang_bmi = raw_features['BANG_BMI']
    bang_neck = raw_features['BANG_Neck']
    bang_gender = raw_features['BANG_Gender']
    
    # Calculate engineered features
    engineered = raw_features.copy()
    
    # 1. Neck_Above_40
    engineered['Neck_Above_40'] = 1 if neck > 40 else 0
    
    # 2. Anthropometric ratios
    engineered['Neck_Height_Ratio'] = neck / height if height > 0 else 0
    engineered['Weight_Height_Ratio'] = weight / height if height > 0 else 0
    
    # 3. ESS Category (encoded)
    ess_cat_str = calculate_ess_category(ess_score)
    engineered['ESS_Category'] = ESS_CATEGORY_MAP.get(ess_cat_str, 2)  # Default to "Normal"
    
    # 4. STOP_Count
    engineered['STOP_Count'] = stop_snore + stop_tired + stop_obs + stop_pressure
    
    # 5. BANG_Total
    engineered['BANG_Total'] = bang_age + bang_bmi + bang_neck + bang_gender
    
    # 6. Airway_Composite
    engineered['Airway_Composite'] = (
        0.3 * bmi +
        0.3 * neck +
        0.2 * alcohol +
        0.2 * snoring
    )
    
    # 7. BMI_Category
    engineered['BMI_Category'] = calculate_bmi_category(bmi)
    
    return engineered


def calculate_age_group(age):
    """Calculate age group from age (used for model input)"""
    if age < 30:
        return 0
    elif age < 50:
        return 1
    else:
        return 2


def calculate_bang_items(age, bmi, neck_circumference, sex):
    """
    Calculate individual BANG items from anthropometric data.
    Returns dict with BANG_Age, BANG_BMI, BANG_Neck, BANG_Gender
    """
    return {
        'BANG_Age': 1 if age > 50 else 0,
        'BANG_BMI': 1 if bmi > 35 else 0,
        'BANG_Neck': 1 if neck_circumference > 40 else 0,
        'BANG_Gender': 1 if sex == 1 else 0  # 1=Male
    }


# ============ SURVEY CALCULATION UTILITIES ============

def calculate_ess_score(responses):
    """
    Calculate Epworth Sleepiness Scale (ESS) score from survey responses.
    responses: list of 8 integers (0-3 for each question)
    Returns: (score, category)
    """
    score = sum(responses)
    
    if score <= 5:
        category = "Low daytime sleepiness (normal)"
    elif score <= 10:
        category = "High daytime sleepiness (normal)"
    elif score in [11, 12]:
        category = "Mild excessive daytime sleepiness"
    elif score in range(13, 16):
        category = "Moderate excessive daytime sleepiness"
    elif score in range(16, 25):
        category = "Severe excessive daytime sleepiness"
    else:
        category = "Invalid"
    
    return score, category


def calculate_berlin_score(category1_items, category2_items, category3_sleepy, bmi):
    """
    Calculate Berlin Questionnaire score.
    Returns: (positive_categories_count, risk_category)
    """
    positive_categories = 0
    
    # Category 1: Snoring and breathing (items 2-6)
    cat1_score = sum(1 for v in category1_items.values() if v)
    if cat1_score >= 2:
        positive_categories += 1
    
    # Category 2: Daytime sleepiness (items 7-9)
    cat2_score = sum(1 for v in category2_items.values() if v)
    if cat2_score >= 2:
        positive_categories += 1
    
    # Category 3: Sleepiness or BMI > 30
    if category3_sleepy or bmi > 30:
        positive_categories += 1
    
    risk_category = "High Risk" if positive_categories >= 2 else "Low Risk"
    
    return positive_categories, risk_category


def calculate_stopbang_score(snoring, tired, observed, pressure, age, neck_circumference, bmi, male):
    """
    Calculate STOP-BANG score.
    Returns: (total_score, risk_category)
    """
    total_score = 0
    stop_score = 0
    
    # STOP questions
    if snoring:
        total_score += 1
        stop_score += 1
    if tired:
        total_score += 1
        stop_score += 1
    if observed:
        total_score += 1
        stop_score += 1
    if pressure:
        total_score += 1
        stop_score += 1
    
    # BANG questions
    if age > 50:
        total_score += 1
    if neck_circumference >= 40.0:
        total_score += 1
    if bmi > 35:
        total_score += 1
    if male:
        total_score += 1
    
    # Determine risk level according to README.md
    if total_score >= 5:
        risk_category = "High Risk"
    elif stop_score >= 2 and (male or bmi > 35 or neck_circumference >= 40.0):
        risk_category = "High Risk"
    elif total_score in range(3, 5):
        risk_category = "Intermediate Risk"
    else:
        risk_category = "Low Risk"
    
    return total_score, risk_category


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'WakeUp Call OSA Prediction API',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


# ============ AUTHENTICATION ENDPOINTS ============

@app.route('/auth/signup', methods=['POST'])
def signup():
    """
    Create a new user account
    
    Expected JSON input:
    {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "password": "secure_password"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['first_name', 'last_name', 'email', 'password']
        missing_fields = [f for f in required_fields if f not in data or not data[f]]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'success': False
            }), 400
        
        first_name = data['first_name'].strip()
        last_name = data['last_name'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Basic validation
        if len(password) < 6:
            return jsonify({
                'error': 'Password must be at least 6 characters long',
                'success': False
            }), 400
        
        if '@' not in email or '.' not in email:
            return jsonify({
                'error': 'Invalid email format',
                'success': False
            }), 400
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Insert user into database
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (first_name, last_name, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (first_name, last_name, email, password_hash))
            
            user_id = cursor.lastrowid
            
            # Generate auth token (valid for 30 days)
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT INTO auth_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, token, expires_at))
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'message': 'Account created successfully',
                'user': {
                    'id': user_id,
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email
                },
                'auth_token': token,
                'expires_at': expires_at.isoformat(),
                'has_survey': False
            }), 201
            
        except sqlite3.IntegrityError:
            return jsonify({
                'error': 'Email already registered',
                'success': False
            }), 409
        
        finally:
            conn.close()
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/auth/login', methods=['POST'])
def login():
    """
    Login with email and password
    
    Expected JSON input:
    {
        "email": "john@example.com",
        "password": "secure_password"
    }
    """
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'error': 'Email and password are required',
                'success': False
            }), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user by email
        cursor.execute('SELECT id, first_name, last_name, email, password_hash FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                'error': 'Invalid email or password',
                'success': False
            }), 401
        
        # Verify password
        if not check_password_hash(user[4], password):
            conn.close()
            return jsonify({
                'error': 'Invalid email or password',
                'success': False
            }), 401
        
        # Generate new auth token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)
        
        cursor.execute('''
            INSERT INTO auth_tokens (user_id, token, expires_at)
            VALUES (?, ?, ?)
        ''', (user[0], token, expires_at))
        
        # Check if user has completed survey
        cursor.execute('''
            SELECT COUNT(*) FROM user_surveys WHERE user_id = ?
        ''', (user[0],))
        has_survey = cursor.fetchone()[0] > 0
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user[0],
                'first_name': user[1],
                'last_name': user[2],
                'email': user[3]
            },
            'auth_token': token,
            'expires_at': expires_at.isoformat(),
            'has_survey': has_survey
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """Logout and invalidate token"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM auth_tokens WHERE token = ?', (token,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/auth/verify', methods=['GET'])
@require_auth
def verify_token():
    """Verify if current token is valid"""
    return jsonify({
        'success': True,
        'user': request.current_user
    }), 200


@app.route('/survey/get-latest', methods=['GET'])
@require_auth
def get_latest_survey():
    """
    Get the latest survey results for the authenticated user
    Returns the most recent survey submission from the database in the same format as submit
    """
    try:
        # Check if this is a guest user
        is_guest = request.current_user.get('is_guest', False)
        if is_guest:
            return jsonify({
                'success': False,
                'message': 'Guest users do not have saved survey data',
                'data': None,
                'is_guest': True
            }), 404
        
        user_id = request.current_user['id']
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, age, sex, height_cm, weight_kg, neck_circumference_cm, bmi,
                   hypertension, diabetes, depression, smokes, alcohol,
                   ess_score, berlin_score, stopbang_score, osa_probability, risk_level, completed_at,
                   sleep_duration_hours, daily_steps
            FROM user_surveys
            WHERE user_id = ?
            ORDER BY completed_at DESC
            LIMIT 1
        ''', (user_id,))
        
        survey = cursor.fetchone()
        conn.close()
        
        if not survey:
            return jsonify({
                'success': False,
                'message': 'No survey data found',
                'data': None
            }), 404
        
        # Extract survey data
        survey_id = survey[0]
        age = survey[1]
        sex_str = survey[2]
        height_cm = survey[3]
        weight_kg = survey[4]
        neck_cm = survey[5]
        bmi = survey[6]
        hypertension = bool(survey[7])
        diabetes = bool(survey[8])
        depression = bool(survey[9])
        smokes = bool(survey[10])
        alcohol = bool(survey[11])
        ess_score = survey[12]
        berlin_score = survey[13]
        stopbang_score = survey[14]
        osa_probability = survey[15]
        risk_level = survey[16]
        # completed_at = survey[17]
        sleep_duration = survey[18] if len(survey) > 18 and survey[18] else 7.0
        daily_steps = survey[19] if len(survey) > 19 and survey[19] else 5000
        
        # Determine score categories
        if ess_score < 8:
            ess_category = "Normal"
        elif ess_score < 12:
            ess_category = "Mild"
        elif ess_score < 16:
            ess_category = "Moderate"
        else:
            ess_category = "Severe"
        
        berlin_category = "High Risk" if berlin_score >= 2 else "Low Risk"
        
        if stopbang_score < 3:
            stopbang_category = "Low Risk"
        elif stopbang_score < 5:
            stopbang_category = "Intermediate Risk"
        else:
            stopbang_category = "High Risk"
        
        # Generate comprehensive recommendations using the recommendation engine
        recommendation = generate_ml_recommendation(
            osa_probability, risk_level, age, bmi, neck_cm,
            hypertension, diabetes, smokes, alcohol,
            ess_score, berlin_score, stopbang_score,
            sleep_duration, daily_steps
        )
        
        # Calculate top risk factors
        top_factors = []
        if bmi >= 30:
            top_factors.append({
                'factor': f'High BMI ({bmi:.1f})',
                'detail': 'Obesity significantly increases OSA risk',
                'impact': 'High',
                'priority': '1'
            })
        if stopbang_score >= 5:
            top_factors.append({
                'factor': f'High STOP-BANG Score ({stopbang_score})',
                'detail': 'Multiple OSA risk factors present',
                'impact': 'High',
                'priority': '2'
            })
        if ess_score >= 11:
            top_factors.append({
                'factor': f'Excessive Daytime Sleepiness (ESS: {ess_score})',
                'detail': 'Significant sleepiness during daytime',
                'impact': 'Medium',
                'priority': '3'
            })
        if age >= 50:
            top_factors.append({
                'factor': f'Age ({age} years)',
                'detail': 'OSA risk increases with age',
                'impact': 'Medium',
                'priority': '4'
            })
        if hypertension:
            top_factors.append({
                'factor': 'Hypertension',
                'detail': 'High blood pressure linked to OSA',
                'impact': 'Medium',
                'priority': '5'
            })
        
        # Use already extracted demographics
        
        # Return in the same format as submit endpoint
        return jsonify({
            'success': True,
            'message': 'Survey data retrieved successfully',
            'data': {
                'success': True,
                'message': 'Survey data retrieved',
                'survey_id': survey_id,
                'demographics': {
                    'age': age,
                    'sex': sex_str,
                    'height_cm': height_cm,
                    'weight_kg': weight_kg,
                    'neck_circumference_cm': neck_cm
                },
                'medical_history': {
                    'hypertension': hypertension,
                    'diabetes': diabetes,
                    'depression': depression,
                    'smokes': smokes,
                    'alcohol': alcohol
                },
                'scores': {
                    'ess': {
                        'score': ess_score,
                        'category': ess_category
                    },
                    'berlin': {
                        'score': berlin_score,
                        'category': berlin_category
                    },
                    'stopbang': {
                        'score': stopbang_score,
                        'category': stopbang_category
                    }
                },
                'prediction': {
                    'osa_probability': round(osa_probability, 3),
                    'risk_level': risk_level,
                    'recommendation': recommendation
                },
                'top_risk_factors': top_factors[:5],  # Return top 5
                'calculated_metrics': {
                    'bmi': round(bmi, 1)
                }
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/survey/submit', methods=['POST'])
@require_auth
def submit_survey():
    """
    Save survey results and generate OSA prediction
    
    Expected JSON input:
    {
        "demographics": {
            "age": 45,
            "sex": "male",
            "height_cm": 175,
            "weight_kg": 85,
            "neck_circumference_cm": 42
        },
        "medical_history": {
            "hypertension": true,
            "diabetes": false,
            "smokes": false,
            "alcohol": true
        },
        "survey_responses": {
            "ess_responses": [2, 1, 2, 1, 2, 1, 2, 1],
            "berlin_responses": {...},
            "stopbang_responses": {...}
        },
        "google_fit": {
            "daily_steps": 8000,
            "sleep_duration_hours": 6.5
        }
    }
    """
    try:
        data = request.get_json()
        user_id = request.current_user['id']
        
        print(f"🔵 === SURVEY SUBMISSION RECEIVED ===")
        print(f"🔵 User ID: {user_id}")
        print(f"🔵 Raw data keys: {list(data.keys())}")
        
        # Extract demographics
        demo = data.get('demographics', {})
        print(f"🔵 Demographics received: {demo}")
        age = demo.get('age', 30)
        sex = 1 if demo.get('sex', 'male').lower() == 'male' else 0
        height_cm = demo.get('height_cm', 170)
        weight_kg = demo.get('weight_kg', 70)
        neck_cm = demo.get('neck_circumference_cm', 37)
        
        # Calculate BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        # Extract medical history
        medical = data.get('medical_history', {})
        hypertension = 1 if medical.get('hypertension', False) else 0
        diabetes = 1 if medical.get('diabetes', False) else 0
        depression = 1 if medical.get('depression', False) else 0
        smokes = 1 if medical.get('smokes', False) else 0
        alcohol = 1 if medical.get('alcohol', False) else 0
        
        # Extract survey responses
        surveys = data.get('survey_responses', {})
        
        # ESS calculation
        ess_responses = surveys.get('ess_responses', [1, 1, 1, 1, 1, 1, 1, 1])
        ess_score, ess_category = calculate_ess_score(ess_responses)
        
        # Extract individual ESS scores for detailed storage
        ess_sitting_reading = ess_responses[0] if len(ess_responses) > 0 else 0
        ess_watching_tv = ess_responses[1] if len(ess_responses) > 1 else 0
        ess_public_sitting = ess_responses[2] if len(ess_responses) > 2 else 0
        ess_passenger_car = ess_responses[3] if len(ess_responses) > 3 else 0
        ess_lying_down_afternoon = ess_responses[4] if len(ess_responses) > 4 else 0
        ess_talking = ess_responses[5] if len(ess_responses) > 5 else 0
        ess_after_lunch = ess_responses[6] if len(ess_responses) > 6 else 0
        ess_traffic_stop = ess_responses[7] if len(ess_responses) > 7 else 0
        
        # Berlin calculation
        berlin_data = surveys.get('berlin_responses', {})
        berlin_cat1 = berlin_data.get('category1', {})
        berlin_cat2 = berlin_data.get('category2', {})
        berlin_cat3_sleepy = berlin_data.get('category3_sleepy', False)
        berlin_score, berlin_category = calculate_berlin_score(berlin_cat1, berlin_cat2, berlin_cat3_sleepy, bmi)
        berlin_score_binary = 1 if berlin_category == "High Risk" else 0
        
        # STOP-BANG calculation
        stopbang_data = surveys.get('stopbang_responses', {})
        snoring = stopbang_data.get('snoring', False)
        tired = stopbang_data.get('tired', False)
        observed = stopbang_data.get('observed_apnea', False)
        pressure = stopbang_data.get('hypertension', hypertension == 1)
        
        stopbang_score, stopbang_category = calculate_stopbang_score(
            snoring, tired, observed, pressure,
            age, neck_cm, bmi, sex == 1
        )
        
        # Extract additional survey fields that aren't in the standard scoring
        snoring_level = surveys.get('snoring_level', 'Unknown')  # "Mild", "Moderate", "Loud", "Very Loud"
        snoring_frequency = surveys.get('snoring_frequency', 'Unknown')  # Frequency of snoring
        snoring_bothers_others = 1 if surveys.get('snoring_bothers_others', False) else 0
        tired_during_day = surveys.get('tired_during_day', 'Unknown')  # Fatigue level
        tired_after_sleep = surveys.get('tired_after_sleep', 'Unknown')  # Post-sleep tiredness
        feels_sleepy_daytime = 1 if surveys.get('feels_sleepy_daytime', tired) else 0
        nodded_off_driving = 1 if surveys.get('nodded_off_driving', False) else 0
        physical_activity_time = surveys.get('physical_activity_time', 'Unknown')  # When they exercise
        
        # Extract Google Fit data
        fit_data = data.get('google_fit', {})
        daily_steps = fit_data.get('daily_steps', 5000)
        average_daily_steps = fit_data.get('average_daily_steps', daily_steps)
        weekly_steps_data = fit_data.get('weekly_steps_data', {})
        weekly_sleep_data = fit_data.get('weekly_sleep_data', {})
        
        print(f"🔵 Google Fit data received:")
        print(f"   Daily steps: {daily_steps}")
        print(f"   Average daily steps: {average_daily_steps}")
        print(f"   Weekly steps data: {len(weekly_steps_data)} days")
        print(f"   Weekly sleep data: {len(weekly_sleep_data)} days")
        
        # Better sleep duration estimation if not provided
        if 'sleep_duration_hours' in fit_data:
            sleep_duration = fit_data['sleep_duration_hours']
            print(f"   Sleep duration from Google Fit: {sleep_duration:.1f}h")
        else:
            # Estimate based on sleep problems (Berlin + ESS scores)
            if berlin_score >= 2:  # High snoring/sleep problems
                sleep_duration = 5.5 + (ess_score / 24) * 2  # 5.5-7.5 hours
            else:
                sleep_duration = 6.5 + (24 - ess_score) / 24 * 1.5  # 6.5-8 hours
            sleep_duration = max(4.0, min(10.0, sleep_duration))
            print(f"   Sleep duration estimated: {sleep_duration:.1f}h")
        
        # Convert Google Fit data to JSON strings for storage
        import json
        weekly_steps_json = json.dumps(weekly_steps_data) if weekly_steps_data else '{}'
        weekly_sleep_json = json.dumps(weekly_sleep_data) if weekly_sleep_data else '{}'
        
        # Estimate additional features
        sleepiness = 1 if ess_score > 10 else 0
        snoring_binary = 1 if snoring else 0
        
        # Estimate activity level from steps (1-5 scale)
        if daily_steps < 3000:
            activity_level = 1  # Sedentary
        elif daily_steps < 6000:
            activity_level = 2  # Low active
        elif daily_steps < 8000:
            activity_level = 3  # Moderate
        elif daily_steps < 10000:
            activity_level = 4  # Active
        else:
            activity_level = 5  # Very active
        
        # Adjust for age and health conditions
        if age > 60: activity_level = max(1, activity_level - 1)
        if hypertension or diabetes: activity_level = max(1, activity_level - 1)
        
        # Estimate sleep quality (inverse of sleepiness)
        sleep_quality = max(1, min(10, 10 - (ess_score // 3)))
        
        # Convert Berlin score to binary (0 = low risk, 1 = high risk)
        berlin_score_binary = 1 if berlin_score >= 2 else 0
        
        # Calculate derived features for new 27-feature model
        age_group = calculate_age_group(age)
        bang_items = calculate_bang_items(age, bmi, neck_cm, sex)
        
        # STOP items from survey responses
        stop_snore = 1 if snoring else 0
        stop_tired = 1 if tired else 0
        stop_obs_apnea = 1 if observed else 0
        stop_pressure = 1 if pressure else 0
        
        # Build raw feature dictionary (before engineering)
        raw_features = {
            'Age': age,
            'Age_Group': age_group,
            'Sex': sex,  # Already binary: 1=male, 0=female
            'Height': height_cm,
            'Weight': weight_kg,
            'BMI': round(bmi, 1),
            'Neck_Circumference': neck_cm,
            'Smokes': smokes,
            'Alcohol': alcohol,
            'Snoring': snoring_binary,
            'Sleepiness': sleepiness,
            'Epworth_Score': ess_score,
            'Berlin_Score': berlin_score_binary,
            'Hypertension': hypertension,
            'Diabetes': diabetes,
            'Depression': depression,
            'STOP_Snore': stop_snore,
            'STOP_Tired': stop_tired,
            'STOP_ObsApnea': stop_obs_apnea,
            'STOP_Pressure': stop_pressure,
            'BANG_Age': bang_items['BANG_Age'],
            'BANG_BMI': bang_items['BANG_BMI'],
            'BANG_Neck': bang_items['BANG_Neck'],
            'BANG_Gender': bang_items['BANG_Gender'],
            'STOPBANG': stopbang_score
        }
        
        # Apply feature engineering to get all 33 features
        input_features = engineer_features(raw_features)
        
        # Make prediction if model is loaded
        osa_probability = 0.0
        certainty = 0.0  # Initialize certainty - will be updated by model prediction
        risk_level = "Unknown"
        recommendation = ""
        
        if model is not None:
            try:
                # DEBUG: Print input features
                print(f"🔍 DEBUG: Input features for ML model ({len(input_features)} features):")
                for feature, value in input_features.items():
                    print(f"  {feature}: {value}")
                
                # Create DataFrame with correct feature order (33 features)
                X_input = pd.DataFrame([{f: input_features[f] for f in FEATURES}])
                print(f"🔍 DEBUG: DataFrame shape: {X_input.shape}")
                print(f"🔍 DEBUG: DataFrame columns: {list(X_input.columns)}")
                
                # Scale numerical features using the loaded scaler
                if scaler is not None:
                    print(f"🔍 DEBUG: Scaling numerical features with scaler")
                    X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])
                else:
                    print(f"⚠️ WARNING: Scaler not loaded, prediction may be inaccurate")
                
                print(f"🔍 DEBUG: DataFrame values (after scaling): {X_input.iloc[0].tolist()}")
                
                # Get prediction - 3-class model (High=0, Intermediate=1, Low=2)
                y_proba = model.predict_proba(X_input)[0]
                y_pred = model.predict(X_input)[0]
                
                # Map prediction to risk level (matching training: High=0, Intermediate=1, Low=2)
                risk_levels = ["High Risk", "Intermediate Risk", "Low Risk"]
                if isinstance(y_pred, (int, np.integer)):
                    risk_level = risk_levels[int(y_pred)]
                    predicted_class_idx = int(y_pred)
                else:
                    risk_level = str(y_pred)
                    predicted_class_idx = list(model.classes_).index(y_pred)
                
                # Get high risk probability (class 0 is High Risk in training)
                osa_probability = float(y_proba[0]) if len(y_proba) > 0 else float(y_proba[predicted_class_idx])
                # Get the certainty of the predicted class (this is what should be shown to user)
                certainty = float(y_proba[predicted_class_idx])
                
                print(f"🔍 DEBUG: Class probabilities: High={y_proba[0]:.3f}, Intermediate={y_proba[1]:.3f}, Low={y_proba[2]:.3f}")
                print(f"🔍 DEBUG: Prediction: {risk_level}, Certainty: {certainty*100:.2f}%")
                print(f"🔍 DEBUG: Sending certainty={certainty:.3f} (will be displayed as {certainty*100:.0f}%)")
                
                # Generate personalized recommendation
                recommendation = generate_ml_recommendation(
                    osa_probability, risk_level, age, bmi, neck_cm, 
                    hypertension, diabetes, smokes, alcohol, 
                    ess_score, berlin_score_binary, stopbang_score,
                    sleep_duration, daily_steps
                )
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        
        # Calculate top risk factors based on actual survey data
        top_factors = calculate_top_risk_factors(input_features, osa_probability)
        print(f"🎯 Top risk factors: {len(top_factors)} factors calculated")
        
        # Check if this is a guest user
        is_guest = request.current_user.get('is_guest', False)
        
        if is_guest:
            # Guest mode - don't save to database, just return results
            print(f"👤 Guest mode - returning results without database save")
            return jsonify({
                'success': True,
                'message': 'Survey processed successfully (Guest Mode)',
                'survey_id': -1,  # No real survey ID for guests
                'demographics': {
                    'age': age,
                    'sex': demo.get('sex', 'male'),
                    'height_cm': height_cm,
                    'weight_kg': weight_kg,
                    'neck_circumference_cm': neck_cm
                },
                'medical_history': {
                    'hypertension': hypertension == 1,
                    'diabetes': diabetes == 1,
                    'depression': depression == 1,
                    'smokes': smokes == 1,
                    'alcohol': alcohol == 1
                },
                'scores': {
                    'ess': {
                        'score': ess_score,
                        'category': ess_category
                    },
                    'berlin': {
                        'score': berlin_score,
                        'category': berlin_category
                    },
                    'stopbang': {
                        'score': stopbang_score,
                        'category': stopbang_category
                    }
                },
                'prediction': {
                    'osa_probability': round(certainty, 3),  # Send certainty (predicted class probability) to display as percentage
                    'risk_level': risk_level,
                    'recommendation': recommendation
                },
                'top_risk_factors': top_factors,
                'calculated_metrics': {
                    'bmi': round(bmi, 1),
                    'estimated_activity_level': activity_level,
                    'estimated_sleep_quality': sleep_quality
                },
                'is_guest': True
            }), 201
        
        # Save to database with all demographics and medical history
        # Check if user already has a survey - if yes, UPDATE instead of INSERT
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            # Check for existing survey
            cursor.execute('SELECT id FROM user_surveys WHERE user_id = ?', (user_id,))
            existing_survey = cursor.fetchone()
            
            if existing_survey:
                # UPDATE existing survey
                survey_id = existing_survey[0]
                print(f"🔄 Updating survey for user {user_id}:")
                print(f"   Age: {age}, Sex: {demo.get('sex', 'male')}, BMI: {bmi:.1f}")
                print(f"   ESS: {ess_score}, Berlin: {berlin_score_binary}, STOP-BANG: {stopbang_score}")
                print(f"   Certainty: {certainty:.3f} ({certainty*100:.1f}%), Risk: {risk_level}")
                
                cursor.execute('''
                    UPDATE user_surveys 
                    SET age = ?, sex = ?, height_cm = ?, weight_kg = ?, neck_circumference_cm = ?, bmi = ?,
                        hypertension = ?, diabetes = ?, depression = ?, smokes = ?, alcohol = ?,
                        ess_score = ?, berlin_score = ?, stopbang_score = ?, 
                        osa_probability = ?, risk_level = ?,
                        daily_steps = ?, average_daily_steps = ?, sleep_duration_hours = ?,
                        weekly_steps_json = ?, weekly_sleep_json = ?,
                        snoring_level = ?, snoring_frequency = ?, snoring_bothers_others = ?,
                        tired_during_day = ?, tired_after_sleep = ?, feels_sleepy_daytime = ?,
                        nodded_off_driving = ?, physical_activity_time = ?,
                        ess_sitting_reading = ?, ess_watching_tv = ?, ess_public_sitting = ?,
                        ess_passenger_car = ?, ess_lying_down_afternoon = ?, ess_talking = ?,
                        ess_after_lunch = ?, ess_traffic_stop = ?,
                        stopbang_snoring = ?, stopbang_tired = ?, stopbang_observed_apnea = ?, stopbang_pressure = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (age, demo.get('sex', 'male'), height_cm, weight_kg, neck_cm, bmi,
                      hypertension, diabetes, depression, smokes, alcohol,
                      ess_score, berlin_score_binary, stopbang_score, certainty, risk_level,
                      daily_steps, average_daily_steps, sleep_duration, weekly_steps_json, weekly_sleep_json,
                      snoring_level, snoring_frequency, snoring_bothers_others,
                      tired_during_day, tired_after_sleep, feels_sleepy_daytime,
                      nodded_off_driving, physical_activity_time,
                      ess_sitting_reading, ess_watching_tv, ess_public_sitting,
                      ess_passenger_car, ess_lying_down_afternoon, ess_talking,
                      ess_after_lunch, ess_traffic_stop,
                      1 if snoring else 0, 1 if tired else 0, 1 if observed else 0, 1 if pressure else 0,
                      user_id))
                
                rows_affected = cursor.rowcount
                print(f"✅ Updated existing survey (ID: {survey_id}, rows affected: {rows_affected}) for user {user_id}")
                
                # Verify the update by reading back
                cursor.execute('''
                    SELECT age, ess_score, berlin_score, stopbang_score, osa_probability, risk_level 
                    FROM user_surveys WHERE user_id = ?
                ''', (user_id,))
                verify = cursor.fetchone()
                if verify:
                    print(f"🔍 Verification - DB now has: Age={verify[0]}, ESS={verify[1]}, Berlin={verify[2]}, STOP-BANG={verify[3]}, OSA={verify[4]:.3f}, Risk={verify[5]}")
            else:
                # INSERT new survey
                cursor.execute('''
                    INSERT INTO user_surveys 
                    (user_id, age, sex, height_cm, weight_kg, neck_circumference_cm, bmi,
                     hypertension, diabetes, depression, smokes, alcohol,
                     ess_score, berlin_score, stopbang_score, osa_probability, risk_level,
                     daily_steps, average_daily_steps, sleep_duration_hours,
                     weekly_steps_json, weekly_sleep_json,
                     snoring_level, snoring_frequency, snoring_bothers_others,
                     tired_during_day, tired_after_sleep, feels_sleepy_daytime,
                     nodded_off_driving, physical_activity_time,
                     ess_sitting_reading, ess_watching_tv, ess_public_sitting,
                     ess_passenger_car, ess_lying_down_afternoon, ess_talking,
                     ess_after_lunch, ess_traffic_stop,
                     stopbang_snoring, stopbang_tired, stopbang_observed_apnea, stopbang_pressure)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, age, demo.get('sex', 'male'), height_cm, weight_kg, neck_cm, bmi,
                      hypertension, diabetes, depression, smokes, alcohol,
                      ess_score, berlin_score_binary, stopbang_score, certainty, risk_level,
                      daily_steps, average_daily_steps, sleep_duration, weekly_steps_json, weekly_sleep_json,
                      snoring_level, snoring_frequency, snoring_bothers_others,
                      tired_during_day, tired_after_sleep, feels_sleepy_daytime,
                      nodded_off_driving, physical_activity_time,
                      ess_sitting_reading, ess_watching_tv, ess_public_sitting,
                      ess_passenger_car, ess_lying_down_afternoon, ess_talking,
                      ess_after_lunch, ess_traffic_stop,
                      1 if snoring else 0, 1 if tired else 0, 1 if observed else 0, 1 if pressure else 0))
                survey_id = cursor.lastrowid
                print(f"✅ Created new survey (ID: {survey_id}) for user {user_id}")
                print(f"   Certainty: {certainty:.3f} ({certainty*100:.1f}%), Risk: {risk_level}")
            
            conn.commit()
            print(f"✅ Database committed - survey data saved successfully")
        except Exception as db_error:
            conn.rollback()
            print(f"❌ Database error: {db_error}")
            raise db_error
        finally:
            conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Survey submitted successfully',
            'survey_id': survey_id,
            'demographics': {
                'age': age,
                'sex': demo.get('sex', 'male'),
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'neck_circumference_cm': neck_cm
            },
            'medical_history': {
                'hypertension': hypertension == 1,
                'diabetes': diabetes == 1,
                'depression': depression == 1,
                'smokes': smokes == 1,
                'alcohol': alcohol == 1
            },
            'scores': {
                'ess': {
                    'score': ess_score,
                    'category': ess_category
                },
                'berlin': {
                    'score': berlin_score,
                    'category': berlin_category
                },
                'stopbang': {
                    'score': stopbang_score,
                    'category': stopbang_category
                }
            },
            'prediction': {
                'osa_probability': round(certainty, 3),  # Send certainty (predicted class probability) to display as percentage
                'risk_level': risk_level,
                'recommendation': recommendation
            },
            'top_risk_factors': top_factors,
            'calculated_metrics': {
                'bmi': round(bmi, 1),
                'estimated_activity_level': activity_level,
                'estimated_sleep_quality': sleep_quality
            }
        }), 201
        
    except Exception as e:
        print(f"❌ ERROR in submit_survey: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False,
            'details': traceback.format_exc()
        }), 500


@app.route('/predict', methods=['POST'])
def predict_osa_risk():
    """
    Predict OSA risk based on user input
    
    Expected JSON input (new 27-feature model):
    {
        "Age": 21,
        "Sex": 0,  # 1=Male, 0=Female
        "Height": 165,  # in cm
        "Weight": 55,  # in kg
        "BMI": 20,  # will be calculated if not provided
        "Neck_Circumference": 34,
        "Hypertension": 0,  # 1=Yes, 0=No
        "Diabetes": 0,
        "Depression": 0,  # 1=Yes, 0=No (new)
        "Smokes": 0,
        "Alcohol": 0,
        "Snoring": 0,  # snoring/loud snoring
        "Sleepiness": 0,  # excessive daytime sleepiness
        "Epworth_Score": 4,
        "Berlin_Score": 0,  # 1=High Risk, 0=Low Risk
        "STOP_Snore": 0,  # Do you snore loudly?
        "STOP_Tired": 0,  # Do you often feel tired?
        "STOP_ObsApnea": 0,  # Has anyone observed you stop breathing?
        "STOP_Pressure": 0,  # Do you have high blood pressure?
        "STOPBANG": 3  # Total STOP-BANG score (0-8)
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check model file.',
            'success': False
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Calculate derived fields if not provided
        age = data.get('Age', 30)
        sex = data.get('Sex', 0)
        height = data.get('Height', 170)
        weight = data.get('Weight', 70)
        neck = data.get('Neck_Circumference', 35)
        
        # Calculate BMI if not provided
        if 'BMI' not in data and height > 0:
            bmi = weight / ((height / 100) ** 2)
            data['BMI'] = round(bmi, 1)
        
        # Calculate Age_Group if not provided
        if 'Age_Group' not in data:
            data['Age_Group'] = calculate_age_group(age)
        
        # Add Height/Weight if not present
        if 'Height' not in data:
            data['Height'] = height
        if 'Weight' not in data:
            data['Weight'] = weight
        
        # Calculate BANG items if not provided (derived from age, bmi, neck, sex)
        bmi = data.get('BMI', 25)
        bang_items = calculate_bang_items(age, bmi, neck, sex)
        for key, value in bang_items.items():
            if key not in data:
                data[key] = value
        
        # Default Depression to 0 if not provided
        if 'Depression' not in data:
            data['Depression'] = 0
        
        # Handle legacy STOPBANG_Total field (rename to STOPBANG)
        if 'STOPBANG_Total' in data and 'STOPBANG' not in data:
            data['STOPBANG'] = data['STOPBANG_Total']
        
        # Default STOP items if not provided (derive from related fields)
        if 'STOP_Snore' not in data:
            data['STOP_Snore'] = data.get('Snoring', 0)
        if 'STOP_Tired' not in data:
            data['STOP_Tired'] = data.get('Sleepiness', 0)
        if 'STOP_ObsApnea' not in data:
            data['STOP_ObsApnea'] = 0  # Can't derive, default to 0
        if 'STOP_Pressure' not in data:
            data['STOP_Pressure'] = data.get('Hypertension', 0)
        
        # Build raw features dict for engineering
        raw_features = {
            'Age': data.get('Age', 30),
            'Age_Group': data.get('Age_Group', calculate_age_group(data.get('Age', 30))),
            'Sex': data.get('Sex', 0),
            'Height': data.get('Height', 170),
            'Weight': data.get('Weight', 70),
            'BMI': data.get('BMI', 25),
            'Neck_Circumference': data.get('Neck_Circumference', 35),
            'Smokes': data.get('Smokes', 0),
            'Alcohol': data.get('Alcohol', 0),
            'Snoring': data.get('Snoring', 0),
            'Sleepiness': data.get('Sleepiness', 0),
            'Epworth_Score': data.get('Epworth_Score', 6),
            'Berlin_Score': data.get('Berlin_Score', 0),
            'Hypertension': data.get('Hypertension', 0),
            'Diabetes': data.get('Diabetes', 0),
            'Depression': data.get('Depression', 0),
            'STOP_Snore': data.get('STOP_Snore', 0),
            'STOP_Tired': data.get('STOP_Tired', 0),
            'STOP_ObsApnea': data.get('STOP_ObsApnea', 0),
            'STOP_Pressure': data.get('STOP_Pressure', 0),
            'BANG_Age': data.get('BANG_Age', bang_items['BANG_Age']),
            'BANG_BMI': data.get('BANG_BMI', bang_items['BANG_BMI']),
            'BANG_Neck': data.get('BANG_Neck', bang_items['BANG_Neck']),
            'BANG_Gender': data.get('BANG_Gender', bang_items['BANG_Gender']),
            'STOPBANG': data.get('STOPBANG', data.get('STOPBANG_Total', 0))
        }
        
        # Apply feature engineering to get all 33 features
        input_features = engineer_features(raw_features)
        
        # Create DataFrame with correct feature order (33 features)
        X_input = pd.DataFrame([{f: input_features[f] for f in FEATURES}])
        
        # Scale numerical features using the loaded scaler
        if scaler is not None:
            X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])
        
        # Get prediction - 3-class model (High=0, Intermediate=1, Low=2)
        y_proba = model.predict_proba(X_input)[0]
        y_pred = model.predict(X_input)[0]
        
        # Map prediction to risk level (matching training: High=0, Intermediate=1, Low=2)
        risk_levels = ["High Risk", "Intermediate Risk", "Low Risk"]
        if isinstance(y_pred, (int, np.integer)):
            risk_level = risk_levels[int(y_pred)]
        else:
            risk_level = str(y_pred)
        
        # Get probability/certainty for the predicted class
        predicted_class_idx = int(y_pred) if isinstance(y_pred, (int, np.integer)) else list(model.classes_).index(y_pred)
        certainty = float(y_proba[predicted_class_idx])
        
        # Get high risk probability (class 0 is High Risk in training)
        high_risk_prob = float(y_proba[0]) if len(y_proba) > 0 else certainty
        
        # Generate comprehensive recommendations
        recommendation = generate_ml_recommendation(
            high_risk_prob, risk_level, 
            data['Age'], data['BMI'], data['Neck_Circumference'],
            data['Hypertension'], data['Diabetes'], data['Smokes'], data['Alcohol'],
            data['Epworth_Score'], data['Berlin_Score'], data.get('STOPBANG', data.get('STOPBANG_Total', 0)),
            data.get('Sleep_Duration', 7.0), data.get('Daily_Steps', 5000)
        )
        
        # Return prediction result
        return jsonify({
            'success': True,
            'prediction': {
                'osa_probability': round(float(high_risk_prob), 3),
                'certainty': round(certainty * 100, 2),
                'osa_class': int(predicted_class_idx),
                'risk_level': risk_level,
                'class_probabilities': {
                    'high': round(float(y_proba[0]), 4),
                    'intermediate': round(float(y_proba[1]), 4) if len(y_proba) > 1 else 0,
                    'low': round(float(y_proba[2]), 4) if len(y_proba) > 2 else 0
                },
                'recommendation': recommendation
            },
            'input_summary': {
                'age': data['Age'],
                'age_group': data['Age_Group'],
                'bmi': data['BMI'],
                'stopbang_score': data.get('STOPBANG', data.get('STOPBANG_Total', 0)),
                'epworth_score': data['Epworth_Score'],
                'sleep_duration': data.get('Sleep_Duration', 7.0),
                'daily_steps': data.get('Daily_Steps', 5000)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/survey/calculate', methods=['POST'])
def calculate_survey_scores():
    """
    Calculate survey scores (ESS, Berlin, STOP-BANG) without prediction.
    
    Expected JSON input:
    {
        "ess_responses": [2, 1, 2, 1, 2, 1, 2, 1],  # 8 values 0-3
        "berlin_category1": {"item2": true, "item3": false, ...},
        "berlin_category2": {"item7": true, ...},
        "berlin_category3_sleepy": true,
        "bmi": 28.5,
        "age": 45,
        "neck_circumference": 37.5,
        "male": true,
        "snoring": true,
        "tired": true,
        "observed_apnea": false,
        "hypertension": true
    }
    """
    try:
        data = request.get_json()
        
        # Calculate ESS
        ess_responses = data.get('ess_responses', [])
        ess_score, ess_category = calculate_ess_score(ess_responses)
        
        # Calculate Berlin
        berlin_cat1 = data.get('berlin_category1', {})
        berlin_cat2 = data.get('berlin_category2', {})
        berlin_cat3_sleepy = data.get('berlin_category3_sleepy', False)
        bmi = data.get('bmi', 25.0)
        berlin_score, berlin_category = calculate_berlin_score(berlin_cat1, berlin_cat2, berlin_cat3_sleepy, bmi)
        
        # Calculate STOP-BANG
        age = data.get('age', 30)
        neck_circumference = data.get('neck_circumference', 37.0)
        male = data.get('male', True)
        snoring = data.get('snoring', False)
        tired = data.get('tired', False)
        observed_apnea = data.get('observed_apnea', False)
        hypertension = data.get('hypertension', False)
        
        stopbang_score, stopbang_category = calculate_stopbang_score(
            snoring, tired, observed_apnea, hypertension,
            age, neck_circumference, bmi, male
        )
        
        # Calculate overall risk
        high_risk_count = 0
        if "Severe" in ess_category or "Moderate" in ess_category:
            high_risk_count += 1
        if berlin_category == "High Risk":
            high_risk_count += 1
        if stopbang_category == "High Risk":
            high_risk_count += 1
        
        overall_risk = "High" if high_risk_count >= 2 else ("Moderate" if high_risk_count >= 1 else "Low")
        
        return jsonify({
            'success': True,
            'survey_scores': {
                'ess': {
                    'score': ess_score,
                    'category': ess_category
                },
                'berlin': {
                    'score': berlin_score,
                    'category': berlin_category
                },
                'stopbang': {
                    'score': stopbang_score,
                    'category': stopbang_category
                },
                'overall_risk_level': overall_risk
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/predict-from-google-fit', methods=['POST'])
def predict_from_google_fit():
    """
    Simplified prediction endpoint that accepts Google Fit data
    and fills in default/estimated values for medical history
    
    Expected JSON input:
    {
        "age": 45,
        "sex": "male",  # or "female"
        "height_cm": 175,
        "weight_kg": 85,
        "neck_circumference_cm": 42,
        "sleep_duration_hours": 6.5,
        "daily_steps": 8000,
        "snores": true,
        "feels_sleepy": true,
        # Optional medical history
        "hypertension": false,
        "diabetes": false,
        "depression": false,
        "smokes": false,
        "alcohol": false,
        "observed_apnea": false
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        
        # Calculate BMI
        height = data['height_cm']
        weight = data['weight_kg']
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        # Convert sex to binary
        sex = 1 if data['sex'].lower() == 'male' else 0
        age = data['age']
        neck = data['neck_circumference_cm']
        
        # Calculate Age Group
        age_group = calculate_age_group(age)
        
        # Medical history fields
        hypertension = 1 if data.get('hypertension', False) else 0
        diabetes = 1 if data.get('diabetes', False) else 0
        depression = 1 if data.get('depression', False) else 0
        smokes = 1 if data.get('smokes', False) else 0
        alcohol = 1 if data.get('alcohol', False) else 0
        
        # Calculate STOP items
        stop_snore = 1 if data.get('snores', False) else 0
        stop_tired = 1 if data.get('feels_sleepy', False) else 0
        stop_obs_apnea = 1 if data.get('observed_apnea', False) else 0
        stop_pressure = hypertension
        
        # Calculate BANG items using helper function
        bang_items = calculate_bang_items(age, bmi, neck, sex)
        
        # Calculate total STOP-BANG score
        stopbang_total = (stop_snore + stop_tired + stop_obs_apnea + stop_pressure +
                        bang_items['BANG_Age'] + bang_items['BANG_BMI'] + 
                        bang_items['BANG_Neck'] + bang_items['BANG_Gender'])
        
        # Snoring and Sleepiness flags
        snoring = stop_snore
        sleepiness = stop_tired
        
        # Estimate Epworth score (simplified)
        epworth_score = 12 if data.get('feels_sleepy', False) else 6
        
        # Estimate Berlin score based on snoring and sleepiness
        berlin_score = 1 if (data.get('snores', False) and data.get('feels_sleepy', False)) else 0
        
        # Build raw feature dictionary
        raw_features = {
            'Age': age,
            'Age_Group': age_group,
            'Sex': sex,
            'Height': height,
            'Weight': weight,
            'BMI': round(bmi, 1),
            'Neck_Circumference': neck,
            'Smokes': smokes,
            'Alcohol': alcohol,
            'Snoring': snoring,
            'Sleepiness': sleepiness,
            'Epworth_Score': epworth_score,
            'Berlin_Score': berlin_score,
            'Hypertension': hypertension,
            'Diabetes': diabetes,
            'Depression': depression,
            'STOP_Snore': stop_snore,
            'STOP_Tired': stop_tired,
            'STOP_ObsApnea': stop_obs_apnea,
            'STOP_Pressure': stop_pressure,
            'BANG_Age': bang_items['BANG_Age'],
            'BANG_BMI': bang_items['BANG_BMI'],
            'BANG_Neck': bang_items['BANG_Neck'],
            'BANG_Gender': bang_items['BANG_Gender'],
            'STOPBANG': stopbang_total
        }
        
        # Apply feature engineering to get all 33 features
        input_features = engineer_features(raw_features)
        
        # Create DataFrame with correct feature order (33 features)
        X_input = pd.DataFrame([{f: input_features[f] for f in FEATURES}])
        
        # Scale numerical features using the loaded scaler
        if scaler is not None:
            X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])
        
        # Get prediction - 3-class model (High=0, Intermediate=1, Low=2)
        y_proba = model.predict_proba(X_input)[0]
        y_pred = model.predict(X_input)[0]
        
        # Map prediction to risk level (matching training: High=0, Intermediate=1, Low=2)
        risk_levels = ["High Risk", "Intermediate Risk", "Low Risk"]
        if isinstance(y_pred, (int, np.integer)):
            risk_level = risk_levels[int(y_pred)]
        else:
            risk_level = str(y_pred)
        
        # Get probability/certainty for the predicted class
        predicted_class_idx = int(y_pred) if isinstance(y_pred, (int, np.integer)) else list(model.classes_).index(y_pred)
        certainty = float(y_proba[predicted_class_idx])
        
        # Get high risk probability (class 0 is High Risk in training)
        high_risk_prob = float(y_proba[0]) if len(y_proba) > 0 else certainty
        
        # Generate recommendation based on risk level
        if risk_level == "Low Risk":
            recommendation = "Your OSA risk is low. Continue maintaining healthy sleep habits."
        elif risk_level == "Intermediate Risk":
            recommendation = "You have intermediate OSA risk. Consider monitoring your sleep patterns and consulting a sleep specialist if symptoms persist."
        else:
            recommendation = "You have high OSA risk. We strongly recommend consulting a sleep specialist soon."
        
        return jsonify({
            'success': True,
            'prediction': {
                'osa_probability': round(float(high_risk_prob), 3),
                'certainty': round(certainty * 100, 2),
                'osa_class': int(predicted_class_idx),
                'risk_level': risk_level,
                'class_probabilities': {
                    'high': round(float(y_proba[0]), 4),
                    'intermediate': round(float(y_proba[1]), 4) if len(y_proba) > 1 else 0,
                    'low': round(float(y_proba[2]), 4) if len(y_proba) > 2 else 0
                },
                'recommendation': recommendation
            },
            'calculated_metrics': {
                'bmi': round(bmi, 1),
                'age_group': age_group,
                'stopbang_score': stopbang_total,
                'estimated_epworth_score': epworth_score,
                'estimated_berlin_score': berlin_score
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/survey/generate-pdf', methods=['POST'])
@require_auth
def generate_pdf_report():
    """
    Generate PDF report from survey data using ReportLab
    Returns PDF file as binary response
    """
    try:
        from pdf_generator import WakeUpCallPDFGenerator
        from io import BytesIO
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from datetime import datetime
        import json
        
        # Check if this is a guest user
        is_guest = request.current_user.get('is_guest', False)
        user_name = f"{request.current_user['first_name']} {request.current_user['last_name']}"
        
        # For guest users, get data from request body; for registered users, get from database
        if is_guest:
            # Get survey data from request body for guest users
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Survey data required for guest PDF generation', 'success': False}), 400
            
            age = data.get('age', 0)
            sex = data.get('sex', 'Unknown')
            height_cm = data.get('height_cm', 0)
            weight_kg = data.get('weight_kg', 0)
            neck_cm = data.get('neck_circumference_cm', 0)
            bmi = data.get('bmi', 0)
            hypertension = data.get('hypertension', 0)
            diabetes = data.get('diabetes', 0)
            smokes = data.get('smokes', 0)
            alcohol = data.get('alcohol', 0)
            ess_score = data.get('ess_score', 0)
            berlin_score = data.get('berlin_score', 0)
            stopbang_score = data.get('stopbang_score', 0)
            osa_probability = data.get('osa_probability', 0)
            risk_level = data.get('risk_level', 'Unknown')
            daily_steps = data.get('daily_steps', 0)
            average_daily_steps = data.get('average_daily_steps', 0)
            sleep_duration_hours = data.get('sleep_duration_hours', 7.0)
            weekly_steps_json = data.get('weekly_steps_json', '{}')
            weekly_sleep_json = data.get('weekly_sleep_json', '{}')
        else:
            user_id = request.current_user['id']
            
            # Get latest survey data from database for registered users
            conn = get_db()
            cursor = conn.cursor()
            
            # First, try to get all columns including new ones
            try:
                cursor.execute('''
                    SELECT age, sex, height_cm, weight_kg, neck_circumference_cm, bmi,
                           hypertension, diabetes, smokes, alcohol,
                           ess_score, berlin_score, stopbang_score, osa_probability, risk_level,
                           daily_steps, average_daily_steps, sleep_duration_hours,
                           weekly_steps_json, weekly_sleep_json,
                           ess_sitting_reading, ess_watching_tv, ess_public_sitting, ess_passenger_car,
                           ess_lying_down_afternoon, ess_talking, ess_after_lunch, ess_traffic_stop,
                           stopbang_snoring, stopbang_tired, stopbang_observed_apnea, stopbang_pressure,
                           physical_activity_time
                    FROM user_surveys
                    WHERE user_id = ?
                    ORDER BY completed_at DESC
                    LIMIT 1
                ''', (user_id,))
                survey = cursor.fetchone()
                has_extended_columns = True
            except Exception as col_error:
                print(f"⚠️ Extended columns not available, using basic query: {col_error}")
                # Fallback to basic columns if new ones don't exist
                cursor.execute('''
                    SELECT age, sex, height_cm, weight_kg, neck_circumference_cm, bmi,
                           hypertension, diabetes, smokes, alcohol,
                           ess_score, berlin_score, stopbang_score, osa_probability, risk_level,
                           daily_steps, average_daily_steps, sleep_duration_hours,
                           weekly_steps_json, weekly_sleep_json
                    FROM user_surveys
                    WHERE user_id = ?
                    ORDER BY completed_at DESC
                    LIMIT 1
                ''', (user_id,))
                survey = cursor.fetchone()
                has_extended_columns = False
            
            conn.close()
            
            if not survey:
                return jsonify({'error': 'No survey data found. Please complete the survey first.', 'success': False}), 404
            
            # Extract data
            age, sex, height_cm, weight_kg, neck_cm, bmi = survey[0:6]
            hypertension, diabetes, smokes, alcohol = survey[6:10]
            ess_score, berlin_score, stopbang_score = survey[10:13]
            osa_probability, risk_level = survey[13:15]
            daily_steps, average_daily_steps, sleep_duration_hours = survey[15:18]
            weekly_steps_json, weekly_sleep_json = survey[18:20] if len(survey) > 19 else ('{}', '{}')
            
            # ESS individual scores (use defaults if not available)
            if has_extended_columns and len(survey) > 27:
                ess_sitting_reading = survey[20] if survey[20] is not None else 0
                ess_watching_tv = survey[21] if survey[21] is not None else 0
                ess_public_sitting = survey[22] if survey[22] is not None else 0
                ess_passenger_car = survey[23] if survey[23] is not None else 0
                ess_lying_down_afternoon = survey[24] if survey[24] is not None else 0
                ess_talking = survey[25] if survey[25] is not None else 0
                ess_after_lunch = survey[26] if survey[26] is not None else 0
                ess_traffic_stop = survey[27] if survey[27] is not None else 0
            else:
                ess_sitting_reading = ess_watching_tv = ess_public_sitting = ess_passenger_car = 0
                ess_lying_down_afternoon = ess_talking = ess_after_lunch = ess_traffic_stop = 0
            
            # STOP-BANG individual responses (use defaults if not available)
            if has_extended_columns and len(survey) > 31:
                stopbang_snoring = bool(survey[28]) if survey[28] is not None else False
                stopbang_tired = bool(survey[29]) if survey[29] is not None else False
                stopbang_observed_apnea = bool(survey[30]) if survey[30] is not None else False
                stopbang_pressure = bool(survey[31]) if survey[31] is not None else False
            else:
                # Estimate from total score if individual values not available
                stopbang_snoring = stopbang_score >= 1
                stopbang_tired = ess_score >= 11
                stopbang_observed_apnea = False
                stopbang_pressure = bool(hypertension)
            
            # Physical activity time (use default if not available)
            physical_activity_time = survey[32] if has_extended_columns and len(survey) > 32 and survey[32] is not None else 'Unknown'
        
        # Initialize ESS variables with defaults if not already set (for non-authenticated users)
        if 'ess_sitting_reading' not in locals():
            ess_sitting_reading = ess_watching_tv = ess_public_sitting = ess_passenger_car = 0
            ess_lying_down_afternoon = ess_talking = ess_after_lunch = ess_traffic_stop = 0
        
        # Use actual ESS individual scores if available, otherwise calculate from total
        if ess_sitting_reading or ess_watching_tv or ess_public_sitting:
            ess_responses = [
                ess_sitting_reading, ess_watching_tv, ess_public_sitting, ess_passenger_car,
                ess_lying_down_afternoon, ess_talking, ess_after_lunch, ess_traffic_stop
            ]
        else:
            # Fallback: distribute total score evenly
            avg_ess = ess_score / 8 if ess_score else 0
            ess_responses = [int(avg_ess)] * 8
        
        # Parse Google Fit JSON data
        import json
        weekly_steps_data = json.loads(weekly_steps_json) if weekly_steps_json else {}
        weekly_sleep_data = json.loads(weekly_sleep_json) if weekly_sleep_json else {}
        
        # Generate charts for PDF
        def generate_shap_chart():
            # Calculate impact scores
            age_impact = 0.75 if age >= 50 else 0.40
            snoring_impact = 0.85 if stopbang_score >= 1 else 0.25
            stopbang_impact = (stopbang_score / 8.0) * 0.9 + 0.1
            
            if neck_cm >= 43:
                neck_impact = 0.90
            elif neck_cm >= 40:
                neck_impact = 0.70
            elif neck_cm >= 37:
                neck_impact = 0.50
            else:
                neck_impact = 0.30
            
            ess_impact = (ess_score / 24.0) * 0.9 + 0.1
            
            factors = [
                ('Age', age_impact),
                ('Snoring', snoring_impact),
                ('STOP-BANG', stopbang_impact),
                ('Neck Circ', neck_impact),
                ('ESS Score', ess_impact)
            ]
            
            # Sort by impact
            factors.sort(key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            names = [f[0] for f in factors]
            values = [f[1] * 100 for f in factors]
            colors = ['#f44336' if v >= 70 else '#ff9800' if v >= 50 else '#4caf50' for v in values]
            
            ax.barh(names, values, color=colors)
            ax.set_xlabel('Impact (%)', fontsize=12)
            ax.set_title('SHAP Analysis - Risk Factor Impact', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)
            
            for i, v in enumerate(values):
                ax.text(v + 2, i, f'{v:.0f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
        
        # Generate weekly steps chart
        def generate_steps_chart(steps_data):
            # Sort by date and take last 7 days
            sorted_data = sorted(steps_data.items(), key=lambda x: x[0])[-7:]
            dates = [d[5:] for d, _ in sorted_data]  # Extract MM-DD
            steps = [s for _, s in sorted_data]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4caf50' if s >= 8000 else '#ff9800' if s >= 5000 else '#f44336' for s in steps]
            bars = ax.bar(dates, steps, color=colors)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Steps', fontsize=12)
            ax.set_title('Weekly Step Count', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(steps) * 1.2 if steps else 15000)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
        
        # Generate weekly sleep chart
        def generate_sleep_chart(sleep_data):
            # Sort by date and take last 7 days
            sorted_data = sorted(sleep_data.items(), key=lambda x: x[0])[-7:]
            dates = [d[5:] for d, _ in sorted_data]  # Extract MM-DD
            hours = [h for _, h in sorted_data]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4caf50' if h >= 7 else '#ff9800' if h >= 6 else '#f44336' for h in hours]
            bars = ax.bar(dates, hours, color=colors)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Hours', fontsize=12)
            ax.set_title('Weekly Sleep Duration', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 10)
            ax.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='Recommended (7h)')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}h',
                       ha='center', va='bottom', fontsize=9)
            
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
        
        # Generate charts
        steps_chart_buffer = generate_steps_chart(weekly_steps_data) if weekly_steps_data else None
        sleep_chart_buffer = generate_sleep_chart(weekly_sleep_data) if weekly_sleep_data else None
        shap_chart_buffer = generate_shap_chart()
        
        # Use actual STOP-BANG responses from survey (not estimates)
        # For guest users, these come from request data; for registered users, from database
        if is_guest:
            snoring = data.get('stopbang_snoring', False)
            tiredness = data.get('stopbang_tired', False) 
            observed_apnea = data.get('stopbang_observed_apnea', False)
        else:
            snoring = stopbang_snoring
            tiredness = stopbang_tired
            observed_apnea = stopbang_observed_apnea
        
        # Calculate BANG components from demographics (these are always calculated)
        bmi_over_35 = bmi > 35
        age_over_50 = age > 50
        neck_large = neck_cm >= 40 if sex == 'Male' else neck_cm >= 35
        gender_male = (sex == 'Male')
        
        # Generate comprehensive recommendations using the recommendation engine
        sex_binary = 1 if sex == 'Male' else 0
        actual_physical_activity = physical_activity_time if not is_guest else data.get('physical_activity_time', None)
        recommendation = generate_ml_recommendation(
            osa_probability, risk_level, age, bmi, neck_cm,
            hypertension, diabetes, smokes, alcohol,
            ess_score, berlin_score, stopbang_score,
            sleep_duration_hours, daily_steps,
            physical_activity_time=actual_physical_activity
        )
        
        # Build data dictionary for PDF generator
        pdf_data = {
            'patient': {
                'name': user_name,
                'age': age,
                'sex': sex,
                'height': f'{height_cm} cm',
                'weight': f'{weight_kg} kg',
                'bmi': bmi,
                'neck_circumference': f'{neck_cm} cm'
            },
            'assessment': {
                'risk_level': risk_level,
                'osa_probability': int(osa_probability * 100),
                'recommendation': recommendation
            },
            'stop_bang': {
                'score': stopbang_score,
                'snoring': snoring,
                'tiredness': tiredness,
                'observed_apnea': observed_apnea,
                'high_blood_pressure': hypertension,
                'bmi_over_35': bmi_over_35,
                'age_over_50': age_over_50,
                'neck_circumference_large': neck_large,
                'gender_male': gender_male
            },
            'epworth_sleepiness_scale': {
                'total_score': ess_score,
                'sitting_reading': ess_responses[0] if ess_responses else 0,
                'watching_tv': ess_responses[1] if ess_responses else 0,
                'public_sitting': ess_responses[2] if ess_responses else 0,
                'passenger_car': ess_responses[3] if ess_responses else 0,
                'lying_down_pm': ess_responses[4] if ess_responses else 0,
                'talking': ess_responses[5] if ess_responses else 0,
                'after_lunch': ess_responses[6] if ess_responses else 0,
                'traffic_stop': ess_responses[7] if ess_responses else 0
            },
            'google_fit': {
                'daily_steps': daily_steps or 0,
                'average_daily_steps': average_daily_steps or 0,
                'sleep_duration_hours': sleep_duration_hours or 0,
                'weekly_steps_chart': steps_chart_buffer,
                'weekly_sleep_chart': sleep_chart_buffer
            },
            'lifestyle': {
                'smoking': smokes,
                'alcohol': alcohol,
                'physical_activity_time': physical_activity_time if not is_guest else data.get('physical_activity_time', 'Unknown')
            },
            'medical_history': {
                'hypertension': hypertension,
                'diabetes': diabetes
            },
            'shap_chart': shap_chart_buffer,
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Generate PDF using ReportLab
        print("📄 Generating PDF report using ReportLab...")
        generator = WakeUpCallPDFGenerator()
        pdf_buffer = generator.generate_pdf(pdf_data)
        
        pdf_size = len(pdf_buffer.getvalue())
        print(f"✅ PDF generated successfully: {pdf_size} bytes")
        
        pdf_buffer.seek(0)
        
        from flask import send_file
        print(f"📤 Sending PDF file to client...")
        response = send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'WakeUpCall_Report_{user_name.replace(" ", "_")}.pdf'
        )
        response.headers['Content-Length'] = pdf_size
        print(f"✅ Response sent with Content-Length: {pdf_size}")
        return response
        
    except ImportError as ie:
        return jsonify({
            'error': f'Missing required library: {str(ie)}. Install with: pip install python-docx matplotlib',
            'success': False
        }), 500
    except Exception as e:
        import traceback
        print(f"❌ PDF Generation Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to generate report: {str(e)}',
            'success': False
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting WakeUp Call OSA Prediction API...")
    print(f"📊 Model loaded: {model is not None}")
    print(f"📏 Scaler loaded: {scaler is not None and hasattr(scaler, 'transform')} (not required for LightGBM)")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌐 Running on port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_bcrypt import Bcrypt
from flask_socketio import SocketIO, emit
import pymongo
from bson import ObjectId
import json
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask_cors import CORS
from sklearn.utils.validation import check_is_fitted
import threading
import eventlet
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-socketio-secret')
jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# Configure CORS explicitly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:4200"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configure Socket.IO with explicit CORS settings
socketio = SocketIO(app, cors_allowed_origins="http://localhost:4200", cors_credentials=True)

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(os.getenv('MONGODB_URI', 'mongodb+srv://decisionmaker707:KZF6njP1WucBCv6r@cluster0.sxwstyz.mongodb.net/app?retryWrites=true&w=majority&appName=Cluster0'))
db = mongo_client['financial_tracker']
users_collection = db['users']
records_collection = db['financial_records']

# Initialize Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize LangChain with OpenAI (for chat only)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.7
)

# LangChain prompt for chat
chat_prompt = PromptTemplate(
    input_variables=["query", "records"],
    template="You are a financial assistant. Based on the user's financial records: {records}, answer the following query: {query}. Provide a concise, helpful response."
)

# LangChain chain for chat
chat_chain = LLMChain(llm=llm, prompt=chat_prompt)

# Tax brackets
TAX_BRACKETS = [
    {'limit': 1000, 'rate': 0.10},
    {'limit': 4000, 'rate': 0.15},
    {'limit': float('inf'), 'rate': 0.25}
]

# Initialize expense prediction model
expense_model = RandomForestRegressor(n_estimators=100)

def calculate_tax(gross_pay):
    tax = 0
    remaining = gross_pay
    for bracket in TAX_BRACKETS:
        if remaining <= 0:
            break
        taxable = min(remaining, bracket['limit'])
        tax += taxable * bracket['rate']
        remaining -= taxable
    return tax

def train_expense_model(expense_data):
    if not expense_data or len(expense_data) < 5:
        return None
    df = pd.DataFrame(expense_data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['category_encoded'] = df['category'].astype('category').cat.codes
    X = df[['month', 'day', 'category_encoded']]
    y = df['amount']
    expense_model.fit(X, y)
    return True

def categorize_expense(description):
    prompt = f"Categorize the following expense description into one of these categories: Food, Transport, Housing, Shopping, Entertainment, Miscellaneous. Description: {description}. Return only the category name."
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_recommendations(category_stats):
    prompt = f"Based on the following expense statistics by category: {json.dumps(category_stats)}, provide 1-2 personalized savings recommendations. Focus on categories with high spending. Return a JSON array of recommendation strings."
    response = gemini_model.generate_content(prompt)
    try:
        return json.loads(response.text.strip())
    except:
        return ["Reduce spending in high-cost categories.", "Review monthly subscriptions for savings."]

def generate_analytics_insights(data):
    prompt = f"Analyze the following financial data: {json.dumps(data)}. Provide 2-3 key insights about spending patterns or trends in a JSON array of strings."
    response = gemini_model.generate_content(prompt)
    try:
        return json.loads(response.text.strip())
    except:
        return ["Consider tracking expenses regularly.", "High spending detected in some categories."]

# MongoDB Change Streams
def watch_records_collection():
    pipeline = [{'$match': {'operationType': {'$in': ['insert', 'update']}}}]
    with records_collection.watch(pipeline) as stream:
        for change in stream:
            eventlet.sleep(0)
            document = change.get('fullDocument', {})
            user_id = document.get('user_id')
            data = document.get('data', {})
            socketio.emit('new_record', {
                'user_id': user_id,
                'data': data,
                'created_at': document.get('created_at', datetime.utcnow()).isoformat()
            }, namespace='/records', room=user_id)

threading.Thread(target=watch_records_collection, daemon=True).start()

# SocketIO events
@socketio.on('connect', namespace='/records')
def handle_connect():
    auth = request.headers.get('Authorization')
    if auth and auth.startswith('Bearer '):
        token = auth.split(' ')[1]
        try:
            user_id = get_jwt_identity()
            emit('connected', {'message': f'Connected as user {user_id}'}, room=user_id)
        except:
            emit('error', {'message': 'Invalid token'}, room=None)
    else:
        emit('error', {'message': 'Authorization header missing'}, room=None)

@socketio.on('join', namespace='/records')
def handle_join(data):
    user_id = data.get('user_id')
    if user_id:
        emit('joined', {'message': f'Joined room {user_id}'}, room=user_id)

# Handle OPTIONS requests for all routes
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return jsonify({}), 200

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    name = data.get('name')
    phone = data.get('phoneNumber')
    password = data.get('password')
    if not all([email, name, phone, password]):
        return jsonify({'message': 'All fields are required'}), 400
    if users_collection.find_one({'email': email}):
        return jsonify({'message': 'User already exists'}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        'email': email,
        'name': name,
        'phone': phone,
        'password': hashed_password,
        'created_at': datetime.utcnow()
    }
    result = users_collection.insert_one(user)
    if result.inserted_id:
        return jsonify({'message': 'User registered successfully'}), 201
    return jsonify({'message': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = users_collection.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'message': 'Invalid credentials'}), 401
    access_token = create_access_token(identity=str(user['_id']))
    return jsonify({'access_token': access_token}), 200

@app.route('/api/calculate-salary', methods=['POST'])
@jwt_required()
def calculate_salary():
    try:
        data = request.get_json()
        if not data or 'rates' not in data:
            return jsonify({'error': 'Missing rates data'}), 400
        total_regular_hours = 0
        total_overtime_hours = 0
        gross_pay = 0
        for entry in data['rates']:
            rate = float(entry['rate'])
            hours = float(entry['hours'])
            if hours <= 40:
                regular_pay = hours * rate
                overtime_pay = 0
            else:
                regular_pay = 40 * rate
                overtime_hours = hours - 40
                overtime_pay = overtime_hours * (rate * 1.5)
            total_regular_hours += min(hours, 40)
            total_overtime_hours += max(0, hours - 40)
            gross_pay += regular_pay + overtime_pay
        tax_amount = calculate_tax(gross_pay)
        net_pay = gross_pay - tax_amount
        monthly_gross = gross_pay * 4.33
        monthly_net = net_pay * 4.33
        yearly_gross = gross_pay * 52
        yearly_net = net_pay * 52
        result = {
            'weekly': {
                'regular_hours': round(total_regular_hours, 1),
                'overtime_hours': round(total_overtime_hours, 1),
                'gross_pay': round(gross_pay, 2),
                'tax_amount': round(tax_amount, 2),
                'net_pay': round(net_pay, 2)
            },
            'monthly': {
                'gross': round(monthly_gross, 2),
                'net': round(monthly_net, 2)
            },
            'yearly': {
                'gross': round(yearly_gross, 2),
                'net': round(yearly_net, 2)
            },
            'created_at': datetime.utcnow().isoformat()
        }
        user_id = get_jwt_identity()
        records_collection.insert_one({
            'user_id': user_id,
            'data': result,
            'created_at': datetime.utcnow()
        })
        socketio.emit('notification', {
            'message': 'New salary calculation saved',
            'user_id': user_id,
            'data': result
        }, namespace='/records', room=user_id)
        return jsonify(result), 200
    except (ValueError, KeyError) as e:
        return jsonify({'error': 'Invalid input data'}), 400

@app.route('/api/analyze-expenses', methods=['POST'])
@jwt_required()
def analyze_expenses():
    try:
        data = request.get_json()
        if not data or 'expenses' not in data:
            return jsonify({'error': 'Missing expenses data'}), 400
        user_id = get_jwt_identity()
        categorized_expenses = []
        for expense in data['expenses']:
            description = expense.get('description', '')
            amount = float(expense.get('amount', 0))
            date = expense.get('date', datetime.now().isoformat())
            category = categorize_expense(description)
            categorized_expenses.append({
                'description': description,
                'amount': amount,
                'date': date,
                'category': category
            })
        df = pd.DataFrame(categorized_expenses)
        category_stats = df.groupby('category')['amount'].agg(['sum', 'count']).to_dict()
        train_expense_model(categorized_expenses)
        recommendations = generate_recommendations(category_stats)
        result = {
            'expenses': categorized_expenses,
            'stats': category_stats,
            'recommendations': recommendations
        }
        records_collection.insert_one({
            'user_id': user_id,
            'data': result,
            'created_at': datetime.utcnow()
        })
        socketio.emit('notification', {
            'message': 'New expenses analyzed',
            'user_id': user_id,
            'data': result
        }, namespace='/records', room=user_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-expenses', methods=['GET'])
@jwt_required()
def predict_expenses():
    try:
        user_id = get_jwt_identity()
        try:
            check_is_fitted(expense_model)
            next_month = datetime.now().month + 1
            if next_month > 12:
                next_month = 1
            categories = ['food', 'transport', 'housing', 'shopping']
            predictions = {}
            for i, category in enumerate(categories):
                day = 15
                X_pred = np.array([[next_month, day, i]])
                predicted_amount = expense_model.predict(X_pred)[0]
                predictions[category] = round(float(predicted_amount), 2)
            return jsonify({
                'month': next_month,
                'predictions': predictions,
                'message': 'Predictions based on trained model.'
            }), 200
        except:
            records = records_collection.find({'user_id': user_id})
            expenses = []
            for record in records:
                if 'expenses' in record['data']:
                    expenses.extend(record['data']['expenses'])
            if not expenses:
                return jsonify({
                    'error': 'No expense records available. Please submit at least one expense via /api/analyze-expenses.'
                }), 400
            df = pd.DataFrame(expenses)
            avg_expenses = df.groupby('category')['amount'].mean().to_dict()
            next_month = datetime.now().month + 1
            if next_month > 12:
                next_month = 1
            predictions = {cat: round(float(avg), 2) for cat, avg in avg_expenses.items()}
            return jsonify({
                'month': next_month,
                'predictions': predictions,
                'message': 'Predictions based on average expenses due to insufficient data for model training.'
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports', methods=['GET'])
@jwt_required()
def get_reports():
    try:
        user_id = get_jwt_identity()
        records = records_collection.find({'user_id': user_id})
        expenses = []
        salaries = []
        for record in records:
            data = record['data']
            if 'expenses' in data:
                for expense in data['expenses']:
                    expense['created_at'] = record['created_at'].isoformat()
                    expenses.append(expense)
            if 'weekly' in data:
                data['created_at'] = record['created_at'].isoformat()
                salaries.append(data)
        expense_df = pd.DataFrame(expenses)
        if not expense_df.empty:
            expense_summary = expense_df.groupby('category')['amount'].agg(['sum', 'count']).to_dict()
        else:
            expense_summary = {}
        return jsonify({
            'expense_summary': expense_summary,
            'salaries': salaries,
            'total_expenses': round(sum(e['amount'] for e in expenses), 2),
            'total_salaries': round(sum(s['weekly']['gross_pay'] for s in salaries), 2)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    try:
        user_id = get_jwt_identity()
        records = records_collection.find({'user_id': user_id})
        expenses = []
        for record in records:
            if 'expenses' in record['data']:
                for expense in record['data']['expenses']:
                    expense['created_at'] = record['created_at'].isoformat()
                    expenses.append(expense)
        expense_df = pd.DataFrame(expenses)
        if expense_df.empty:
            return jsonify({'trends': {}, 'insights': []}), 200
        expense_df['date'] = pd.to_datetime(expense_df['date'])
        monthly_trends = expense_df.groupby([expense_df['date'].dt.to_period('M'), 'category'])['amount'].sum().unstack().fillna(0).to_dict()
        insights = generate_analytics_insights({'monthly_trends': monthly_trends})
        return jsonify({
            'trends': monthly_trends,
            'insights': insights
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-data', methods=['POST'])
@jwt_required()
def save_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        user_id = get_jwt_identity()
        records_collection.insert_one({
            'user_id': user_id,
            'data': data,
            'created_at': datetime.utcnow()
        })
        socketio.emit('notification', {
            'message': 'New data saved',
            'user_id': user_id,
            'data': data
        }, namespace='/records', room=user_id)
        return jsonify({'message': 'Data saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to save data'}), 500

@app.route('/api/get-records', methods=['GET'])
@jwt_required()
def get_records():
    try:
        user_id = get_jwt_identity()
        records = records_collection.find({'user_id': user_id})
        records_list = [record['data'] for record in records]
        return jsonify(records_list), 200
    except Exception as e:
        return jsonify({'error': 'Failed to read records'}), 500

@app.route('/api/chat', methods=['POST'])
@jwt_required()
def chat():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        user_id = get_jwt_identity()
        records = records_collection.find({'user_id': user_id})
        records_list = [record['data'] for record in records]
        response = chat_chain.run(query=user_query, records=json.dumps(records_list))
        socketio.emit('chat_message', {
            'query': user_query,
            'response': response,
            'user_id': user_id
        }, namespace='/records', room=user_id)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
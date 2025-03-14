# Standard Library Imports
import os
import base64
import io
from datetime import datetime
from collections import Counter
import re

# Third-Party Imports
from flask import Flask, render_template, redirect, url_for, flash, request, Response, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from flask_migrate import Migrate
from wtforms import StringField, PasswordField, SubmitField, DecimalField
from wtforms.validators import DataRequired, Length, EqualTo, NumberRange
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv

# Data Processing and Machine Learning
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Image Processing and OCR
from PIL import Image
import pytesseract
import matplotlib

matplotlib.use('Agg')  # Prevent GUI errors in Flask

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Change to PostgreSQL/MySQL if needed
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.init_app(app)
migrate = Migrate(app,db)

load_dotenv()  # Load variables from .env


google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    offline=True,  # To get refresh token
    scope=["profile", "email"]
)
app.register_blueprint(google_bp, url_prefix="/login")



UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

# ---- MODELS ----
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=True)
    balance = db.Column(db.Float, default=1000.0)
    email = db.Column(db.String(120), unique=True, nullable=True)

    def get_id(self):
        return str(self.id)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)  # Deposit or Withdraw

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---- FORMS ----
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class TransactionForm(FlaskForm):
    amount = DecimalField('Amount', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Submit')

# ---- ROUTES ----
@app.route("/")
def home():
    return render_template("home.html")

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            flash("Username already taken.", "danger")
            return redirect(url_for("register"))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password, balance=1000.0)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form=form)

# Normal Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("dashboard"))
        if user:
            if user.password is None:
                flash("This account is associated with Google OAuth. Please log in with Google.","info")
                return redirect(url_for(google_login))

        flash("Login failed! Check your credentials.", "danger")

    return render_template("login.html", form=form)

# Google Login Route
@app.route('/google-login')
def google_login():
    if not google.authorized:
        return redirect(url_for('google.login'))  # Redirect to Google OAuth login
    return redirect(url_for('dashboard'))  # If already authorized, redirect to dashboard

# Google OAuth Callback Route
@app.route('/login/callback')
def google_callback():
    if not google.authorized:
        flash("Failed to authorize with Google.", "danger")
        return redirect(url_for('login'))

    # Fetch user info from Google
    resp = google.get("/oauth2/v1/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", "danger")
        return redirect(url_for('login'))

    user_info = resp.json()

    # Check if user exists in your database
    user = User.query.filter_by(email=user_info['email']).first()

    if not user:
        # Create a new user with Google info
        user = User(
            username=user_info.get('name', user_info['email']),  # Use email as username if name is not provided
            email=user_info['email'],
            password=None,  # No password for OAuth users
            balance=1000.0
        )
        db.session.add(user)
        db.session.commit()

    # Log the user in
    login_user(user)
    flash("Logged in successfully with Google!", "success")
    return redirect(url_for('dashboard'))

# Dashboard Route
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    form = TransactionForm()
    user = User.query.filter_by(id=current_user.id).first()

    if not user:
        flash("User not found!", "danger")
        return redirect(url_for("login"))

    balance = user.balance

    # Fetch transaction history for the current user
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.id.desc()).all()

    if form.validate_on_submit():
        amount = float(form.amount.data)
        transaction_type = request.form.get("transaction_type")

        if transaction_type == "Deposit":
            user.balance += amount
            # Create a new transaction record for deposit
            transaction = Transaction(user_id=current_user.id, amount=amount, type="Deposit")
        elif transaction_type == "Withdraw":
            if user.balance >= amount:
                user.balance -= amount
                # Create a new transaction record for withdrawal
                transaction = Transaction(user_id=current_user.id, amount=amount, type="Withdraw")
            else:
                flash("Insufficient funds!", "danger")
                return redirect(url_for("dashboard"))

        # Add the new transaction to the database
        db.session.add(transaction)
        db.session.commit()

        # flash(f"{transaction_type} of ₹{amount} successful!", "success")
        return redirect(url_for("dashboard"))

    return render_template("dashboard.html", form=form, balance=user.balance, transactions=transactions)

# Logout Route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))
# ---- LOAN CHECKER ----
Loan = [
    {
        "name": "SBI Student Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "HDFC Credila Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Starts at 9.25% p.a.",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "PNB Udaan Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "Axis Bank Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Starts at 13.70% p.a.",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "Bank of Baroda Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to BOB’s RLLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "Canara Bank Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "ICICI Bank Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Starts at 9.50% p.a.",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "Union Bank of India Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "IDBI Bank Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    },
    {
        "name": "Indian Bank Education Loan",
        "eligibility": "Admission to a recognized institution in India or abroad",
        "interest_rate": "Linked to MCLR",
        "criteria": lambda s: s['admission_status'] == 'confirmed' and s['institution'] in ['India', 'Abroad']
    }
]

@app.route('/loan_checker')
def loan_checker():
    return render_template('loan_checker.html')

@app.route('/check-loan', methods=['POST'])
def check_loan():
    # Collect student data from the form
    student = {
        "name": request.form['name'],
        "age": int(request.form['age']),
        "gender": request.form['gender'].lower(),
        "percentage": float(request.form['percentage']),
        "part_time_job": request.form['part_time'],
        "income": int(request.form['income']),
        "disability": request.form.get('disability', 'no'),
        "sports_quota": request.form.get('sports_quota', 'no'),
        "orphan": request.form.get('orphan', 'no'),
        "minority": request.form.get('minority', 'no'),
        "admission_status": request.form.get('admission_status', 'confirmed'),  # Required for criteria
        "institution": request.form.get('institution', 'India')  # Required for criteria
    }

    eligible_loans = []

    # Iterate over the Loan list directly (since it's a list, not a model)
    for loan in Loan:
        try:
            # Use the criteria lambda function to check eligibility
            if loan['criteria'](student):
                eligible_loans.append(loan)
        except KeyError as e:
            flash(f"Missing required field for loan eligibility check: {str(e)}", "danger")
            return redirect(url_for('loan_checker'))

    return render_template('results.html', student=student, loans=eligible_loans)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    extracted_text = db.Column(db.Text, nullable=True)
    cluster_label = db.Column(db.Integer, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    document_type = db.Column(db.String(50), nullable=False)  # structured, semi_structured, unstructured
    result = db.Column(db.Text, nullable=True)  # Store visualization or processed data

    def __repr__(self):
        return f"<Document {self.filename}>"
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_ocr(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def generate_random_dataset():
    data = np.random.rand(10, 4)  # 10 rows, 4 columns
    columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    return pd.DataFrame(data, columns=columns)

def perform_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    return df

def generate_csv_visualization(csv_path):
    # Read CSV and generate a simple line chart
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    df.plot(kind='line')
    plt.title("Line Chart Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        document_type = request.form.get("document_type")

        if document_type == "structured":
            if 'file' not in request.files:
                flash("Please upload a file", "danger")
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                flash("Please upload a valid image file", "danger")
                return redirect(request.url)

            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform OCR
            extracted_text = perform_ocr(file_path)

            # Save to database
            new_document = Document(
                filename=filename,
                extracted_text=extracted_text,
                document_type="structured"
            )
            db.session.add(new_document)
            db.session.commit()

            flash("Structured document uploaded and processed!", "success")
            return redirect(url_for('uploaded_files'))  # Redirect to structured page

        elif document_type == "semi_structured":
            if 'file' not in request.files:
                flash("Please upload a file", "danger")
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '' or not file.filename.endswith('.csv'):
                flash("Please upload a valid CSV file", "danger")
                return redirect(request.url)

            # Save the CSV file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Convert date columns to datetime objects (if applicable)
            # Example: If the first column contains dates, convert it to datetime
            date_column = df.columns[0]  # Assuming the first column is the date column
            df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')

            # Drop rows with invalid dates (if any)
            df = df.dropna(subset=[date_column])

            # Identify numeric columns (exclude the date column)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Generate insights (only for numeric columns)
            insights = {
                'columns': df.columns.tolist(),
                'rows': df.shape[0],
                'average': df[numeric_columns].mean().to_dict(),  # Only numeric columns
                'min': df[numeric_columns].min().to_dict(),      # Only numeric columns
                'max': df[numeric_columns].max().to_dict(),      # Only numeric columns
                'trend': df[numeric_columns[0]].diff().mean()    # Only numeric columns
            }

            # Generate visualization
            plot_url = generate_csv_visualization(file_path)

            # Save to database
            new_document = Document(
                filename=filename,
                document_type="semi_structured",
                result=plot_url  # Store the visualization URL
            )
            db.session.add(new_document)
            db.session.commit()

            flash("Semi-structured document uploaded and processed!", "success")
            return render_template('uploaded_semi_structured.html', filename=filename, data=df.to_dict(), insights=insights, plot_url=plot_url)

        elif document_type == "unstructured":
            # Generate random dataset
            df = generate_random_dataset()
            # Perform clustering
            df = perform_clustering(df)
            # Save to file
            filename = f"random_dataset_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            df.to_csv(file_path, index=False)

            # Save to database
            new_document = Document(
                filename=filename,
                document_type="unstructured"
            )
            db.session.add(new_document)
            db.session.commit()

            flash("Unstructured dataset generated and clustered!", "success")
            return redirect(url_for('unstructured'))  # Redirect to unstructured page

    return render_template('upload.html')

@app.route('/uploaded_files')
def uploaded_files():
    documents = Document.query.order_by(Document.uploaded_at.desc()).all()
    processed_data = []

    for doc in documents:
        data = {'id': doc.id, 'filename': doc.filename, 'document_type': doc.document_type}
        if doc.document_type == "structured":
            data['result'] = doc.extracted_text
        elif doc.document_type == "semi_structured":
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
            data['result'] = generate_csv_visualization(file_path)
        elif doc.document_type == "unstructured":
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
            df = pd.read_csv(file_path)
            data['result'] = df.to_html(classes='table table-striped')
        processed_data.append(data)

    return render_template('uploaded_files.html', documents=processed_data)

@app.route('/unstructured')
def unstructured():
    # Predefined subset of the Iris dataset (Petal Length and Petal Width)
    data = {
        'Petal Length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
        'Petal Width': [0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.3, 0.2, 0.2, 0.1]
    }
    df = pd.DataFrame(data)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)

    # Convert the DataFrame to a dictionary for rendering
    dataset = df.to_dict(orient='list')

    # Pass the dataset and clustering results to the template
    return render_template('unstructured.html', dataset=dataset, clusters=df['Cluster'].tolist())

@app.route('/visualization')
def visualization():
    documents = Document.query.all()
    if not documents:
        flash("No documents uploaded yet!", "warning")
        return redirect(url_for('uploaded_files'))

    # Extract text from all structured documents
    all_text = " ".join([doc.extracted_text for doc in documents if doc.extracted_text])

    if not all_text.strip():
        flash("No text available for visualization!", "warning")
        return redirect(url_for('uploaded_files'))

    # Count the most common words (excluding stop words)
    words = all_text.lower().split()
    common_words = [word for word in words if word not in ['the', 'is', 'and', 'to', 'of', 'a']]  # Basic stop words
    word_counts = Counter(common_words)
    most_common = word_counts.most_common(10)  # Top 10 words

    # Bar Chart for Word Frequency
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[word[0] for word in most_common], y=[word[1] for word in most_common], palette="viridis")
    plt.title("Top 10 Most Common Words in Extracted Text")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    bar_chart_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # New Visualization: Histogram of Numeric Values (e.g., amounts from bank statements)
    numeric_values = re.findall(r'\b\d+\.\d+\b', all_text)  # Extract numeric values (e.g., 123.45)
    if numeric_values:
        numeric_values = list(map(float, numeric_values))  # Convert to floats
        plt.figure(figsize=(10, 5))
        sns.histplot(numeric_values, bins=10, kde=True, color='blue')
        plt.title("Distribution of Numeric Values in Extracted Text")
        plt.xlabel("Numeric Values")
        plt.ylabel("Frequency")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        histogram_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
    else:
        histogram_b64 = None  # No numeric values found

    return render_template('visualization.html', bar_chart_b64=bar_chart_b64, histogram_b64=histogram_b64)

# ---- DATABASE INITIALIZATION ----
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
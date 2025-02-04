# Import necessary libraries
import re
import joblib
import pandas as pd
import nltk
from flask import Flask, render_template, request, redirect, url_for, session
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize Flask app
app = Flask(__name__)
app.secret_key ="c39d86c0ba4945e098a7672d1d3f4e12"  # Required for session management

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preload stopwords globally
STOPWORDS = set(stopwords.words('english'))

# Load the dataset (ensure file path is correct)
df = pd.read_csv('Reviews.csv')

# Drop rows with missing or empty 'Text' values
df = df.dropna(subset=['Text'])

# Ensure all 'Text' values are strings
df['Text'] = df['Text'].astype(str)

# Simplify sentiment classification: Positive if Score >= 4, else Negative
df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')

# Keep only required columns
df = df[['Text', 'Sentiment']]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function: clean and tokenize the text
def preprocess_text(text):
    try:
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
        return " ".join(tokens)
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return ""

# Debugging: Process only the first few rows initially
df = df.head(100)  # Limit dataset for debugging

# Apply preprocessing to the dataset
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# Separate features and labels
X = df['Processed_Text']
y = df['Sentiment']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=10000)  # Adjust max_features as needed

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Initialize Naive Bayes classifier
model = MultinomialNB()

# Train the model using the transformed training data
model.fit(X_train_vec, y_train)

# Evaluate the model's performance
accuracy = model.score(vectorizer.transform(X_test), y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to predict sentiment of new reviews (from user input)
def predict_sentiment(text):
    # Preprocess and vectorize the input text
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict sentiment
    prediction = model.predict(vectorized_text)
    return prediction[0]  # 'Positive' or 'Negative'

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

# Route for admin login
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin.html', error="Invalid credentials")
    return render_template('admin.html')

# Admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return "<h1>Welcome, Admin!</h1> <a href='/admin/logout'>Logout</a>"

# Admin logout route
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

# Home route to render the index.html
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review text from the form
        review_text = request.form['review_text']
        
        # Get the predicted sentiment
        sentiment = predict_sentiment(review_text)
        
        # Render the result on the page
        return render_template('index.html', sentiment=sentiment, review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True)

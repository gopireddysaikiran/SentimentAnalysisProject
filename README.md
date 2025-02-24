# 🌟 Sentiment Analysis Web App

A Flask-based web application for sentiment analysis using Naïve Bayes classification. Users can enter text reviews, and the model will determine whether the sentiment is positive or negative.

---

## 🚀 Features

✅ **AI-Powered Sentiment Detection** – Classifies reviews as Positive or Negative  
✅ **User-Friendly Web Interface** – Built using Flask and HTML/CSS  
✅ **Machine Learning Model** – Uses Naïve Bayes for text classification  
✅ **Admin Panel** – Secure admin login for dashboard access  
✅ **Session-Based User Management** – Keeps track of user activity  

---

## 🏗️ Tech Stack

| Component    | Technology Used           |
|--------------|--------------------------|
| Backend      | Flask (Python)           |
| Frontend     | HTML, CSS, Bootstrap     |
| Machine Learning | Scikit-Learn, NLTK, Naïve Bayes |
| Database     | CSV-based storage        |
| Deployment   | Local / Cloud            |

---

## 📂 Project Structure

```
sentiment_analysis/
│── app.py              # Main Flask app
│── index.html          # Frontend template
│── Reviews.csv         # Dataset (user reviews)
│── sentiment_model.pkl # Trained ML model
│── vectorizer.pkl      # CountVectorizer for text preprocessing
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/sentiment_analysis.git
cd sentiment_analysis
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4️⃣ Run the Application

```bash
python app.py
```

Access the Web App at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🎨 UI Preview

(Add screenshots or GIFs of the app's interface here)

---

## 📊 How It Works

1️⃣ User submits a review in the web interface  
2️⃣ Text preprocessing (stopword removal, lemmatization) is applied  
3️⃣ **Vectorization** – Text is converted into numerical format  
4️⃣ **Naïve Bayes Classifier** predicts **Positive** or **Negative**  
5️⃣ Result is displayed on the web page  

---

## 🔑 Admin Panel

🔹 Admin can log in at `/admin`  
🔹 Dashboard access for managing model predictions  
🔹 Logout option included  

---

## 🏆 Model Accuracy

🎯 The trained **Naïve Bayes Model** achieves high accuracy, making it reliable for sentiment analysis tasks.

- **Accuracy:** ~85-90% on test data

---

## 📜 License

👤 This project is open-source. Feel free to modify and improve it!

---

## 📌 Notes

### 📊 Dataset

We use a publicly available sentiment analysis dataset from Kaggle. You can use any dataset of your choice, such as:

- **Amazon Reviews Data**  
- **Twitter Sentiment Analysis**  

### 💾 To use your dataset:

1. Download the CSV file from Kaggle.
2. Place it in your project folder and rename it as `Reviews.csv`.
3. Make sure the dataset has at least two columns:
   - **Text** (User review)
   - **Score** or **Sentiment** (Label: Positive/Negative)

You can modify the preprocessing steps in `app.py` if needed.

---

Happy Coding! 🚀

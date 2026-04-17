
# 📊 Text Mining Project: Spam Classification with Explainability

## 📌 Overview

This project implements a **text mining-based spam detection system** that classifies SMS messages as **Spam 🚨** or **Not Spam ✅**.
It also provides **explainability** by highlighting the important words that influenced the prediction.

---

## 🎯 Objectives

* Classify text messages into spam or non-spam
* Apply text preprocessing techniques
* Convert text into numerical features using TF-IDF
* Train a machine learning model for classification
* Provide explanation for predictions

---

## 🧠 Techniques Used

* Text Preprocessing (Cleaning, Stopword Removal)
* TF-IDF Vectorization
* Machine Learning (Multinomial Naive Bayes)
* Explainable AI (Feature Importance)

---

## 📂 Dataset

* **SMS Spam Collection Dataset**
* Contains labeled messages:

  * `ham` → Not Spam
  * `spam` → Spam

---

## ⚙️ Project Structure

```
text-mining-spam-classifier/
│
├── data/
│   └── spam.csv
│
├── src/
│   ├── train_model.py
│   └── main.py
│
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

### 1. Data Preprocessing

* Convert text to lowercase
* Remove special characters and numbers
* Remove stopwords

### 2. Feature Extraction

* TF-IDF converts text into numerical vectors

### 3. Model Training

* Multinomial Naive Bayes learns patterns in spam messages

### 4. Prediction

* Takes user input
* Classifies as spam or not spam

### 5. Explainability

* Displays important words contributing to prediction

---

## ▶️ How to Run

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

### Step 2: Train model (only once)

```
python src/train_model.py
```

### Step 3: Run application

```
python src/main.py
```

---

## 🧪 Example

**Input:**

```
You have won a free prize! Click now
```

**Output:**

```
Prediction: SPAM 🚨  
Confidence: 92%  

Important words:
free → high importance  
won → high importance  
```

---

## 📈 Key Features

* Interactive user input
* Fast predictions (model saved using pickle)
* Explainable outputs
* Improved accuracy using n-grams

---

## ⚠️ Limitations

* Model depends on dataset quality
* May not detect modern phishing messages perfectly
* Uses simple ML model (not deep learning)

---

## 🔮 Future Improvements

* Use advanced models (LSTM, BERT)
* Add web interface (Streamlit)
* Improve dataset with modern spam examples
* Deploy as an API

---

## 👨‍💻 Author

**Abhiram Aravind**
Text Mining Lab Project

---

## 🏁 Conclusion

This project demonstrates how text mining techniques can be applied to real-world problems like spam detection.
It combines machine learning with explainability to provide meaningful insights into predictions.

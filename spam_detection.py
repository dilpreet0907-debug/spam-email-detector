# =========================
# STEP 1: Import Libraries
# =========================

import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# =========================
# STEP 2: Load Dataset
# =========================

data = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", header=None)
data.columns = ['label', 'text']

print("Dataset Loaded Successfully")
print(data.head())


# Keep only required columns
# STEP 2: Load Dataset
data = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", header=None)
data.columns = ['label', 'text']

print("Dataset Loaded Successfully")
print(data.head())



# =========================
# STEP 3: Text Preprocessing
# =========================

def clean_text(text):
    text = text.lower()                       # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

data['text'] = data['text'].apply(clean_text)


# =========================
# STEP 4: Feature Extraction (TF-IDF)
# =========================

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])

y = data['label']


# =========================
# STEP 5: Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# STEP 6: Train Naive Bayes Model
# =========================

model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("\nModel Evaluation Results")
print("-" * 40)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


print("\nModel Trained Successfully")


# =========================
# STEP 7: Model Evaluation
# =========================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# STEP 8: Test with New Email
# =========================

def predict_email(email_text):
    email_text = clean_text(email_text)
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return prediction[0]


# Example testing
test_email = "Congratulations! You have won a free prize. Click now"
result = predict_email(test_email)

print("\nTest Email:", test_email)
print("Prediction:", result)

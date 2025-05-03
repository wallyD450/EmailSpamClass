from django.shortcuts import render

# Create your views here.

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from .forms import MessageForm

# Load dataset
email_spam = pd.read_csv("emails.csv")

# Simple preprocessing: lowercase and replace URLs
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+', 'URL', text)  # Replace URLs
    return text

# Apply preprocessing
email_spam['text'] = email_spam['text'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
vectorized_X = vectorizer.fit_transform(email_spam['text'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(vectorized_X, email_spam['spam'], test_size=0.2, random_state=42)

# Train model
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Function to predict spam/ham
def EmailSpam(message):
    processed_message = preprocess_text(message)
    message_vector = vectorizer.transform([processed_message])
    prediction = model.predict(message_vector)
    return "Spam mail" if prediction[0] == 1 else "Ham mail"

# Test examples
def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = EmailSpam(message)
    else:
        form = MessageForm()
    return render(request, 'home.html', {'form': form, 'result': result})

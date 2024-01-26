import nltk
nltk.download('all')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
import re
import string
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#loading the data set
data=pd.read_csv('sentiment-analysis.csv')
data.head()
#separating the data into columns
data = data['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'].str.split(', ', expand=True)

data.columns = ['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']
print('There are {} rows and {} columns in data.'.format(data.shape[0], data.shape[1]))

data.head(10)

#Data Cleaning
data.isnull().sum()
data.dropna(inplace=True)
data.head(10)

#there are some leading and trailing spaces in 'Date/Time' column, so let's trim them
data['Date/Time'] = data['Date/Time'].str.strip()

data[['Date', 'Time']] = data['Date/Time'].str.split(' ', expand=True)

data.drop(columns=['Date/Time'], inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time

#Data Distribution
sns.countplot(x='Sentiment', data=data, palette= ['green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of Positive and Negative Reviews')
plt.show()
data['Sentiment'] = data['Sentiment'].str.strip()
data['Sentiment'] = data['Sentiment'].str.lower()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

review_length_negative = data[data['Sentiment'] == 'negative']['Text'].str.len()
axes[0, 0].hist(review_length_negative, color='red')
axes[0, 0].set_title("Negative Sentiment - Characters")

review_length_positive = data[data['Sentiment'] == 'positive']['Text'].str.len()
axes[0, 1].hist(review_length_positive, color='green')
axes[0, 1].set_title("Positive Sentiment - Characters")

review_length_negative_words = data[data['Sentiment'] == 'negative']['Text'].str.split().map(lambda x: len(x))
axes[1, 0].hist(review_length_negative_words, color='red')
axes[1, 0].set_title("Negative Sentiment - Words")

review_length_positive_words = data[data['Sentiment'] == 'positive']['Text'].str.split().map(lambda x: len(x))
axes[1, 1].hist(review_length_positive_words, color='green')
axes[1, 1].set_title("Positive Sentiment - Words")

plt.tight_layout()

plt.show()

data['DayOfWeek'] = data['Date'].dt.day_name()
custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['DayOfWeek'] = pd.Categorical(data['DayOfWeek'], categories=custom_order, ordered=True)

sns.countplot(x='DayOfWeek', hue='Sentiment', data=data, palette= ['green', 'red'])

plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.title('Sentiment Distribution by Weekday')
plt.legend(title='Sentiment', labels=['Positive', 'Negative'], bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
sns.countplot(y='Source', hue='Sentiment', data=data, palette=['green', 'red'])
plt.xlabel('Count')
plt.ylabel('Source')
plt.title('Positive and Negative Reviews by Source')
plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), labels=['Positive', 'Negative'])
plt.show()

sns.countplot(y='Location', hue='Sentiment', data=data, palette=['green', 'red'])
plt.xlabel('Count')
plt.ylabel('Source')
plt.title('Positive and Negative Reviews by Location')
plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), labels=['Positive', 'Negative'])
plt.show()



#ALGORITMS
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preparing the data
X = data['Text']
y = data['Sentiment']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_tfidf, y_train)

# Predictions
svm_predictions = svm_classifier.predict(X_test_tfidf)

# Evaluating SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print("\nClassification Report for SVM:\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix for SVM:\n", confusion_matrix(y_test, svm_predictions))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predictions
rf_predictions = rf_classifier.predict(X_test_tfidf)

# Evaluating Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("\nClassification Report for Random Forest:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix for Random Forest:\n", confusion_matrix(y_test, rf_predictions))


import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plotting Confusion Matrix for SVM
plot_confusion_matrix(y_test, svm_predictions, 'SVM Confusion Matrix', ['negative', 'positive'])

# Plotting Confusion Matrix for Random Forest
plot_confusion_matrix(y_test, rf_predictions, 'Random Forest Confusion Matrix', ['negative', 'positive'])

# Plotting Accuracy Comparison
models = ['SVM', 'Random Forest']
accuracies = [svm_accuracy, rf_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['green', 'blue'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison - SVM vs Random Forest')
plt.ylim(0, 1)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plotting Confusion Matrix for SVM (Test Set)
plot_confusion_matrix(y_test, svm_predictions, 'SVM Test Set Confusion Matrix', ['negative', 'positive'])

# Plotting Confusion Matrix for Random Forest (Test Set)
plot_confusion_matrix(y_test, rf_predictions, 'Random Forest Test Set Confusion Matrix', ['negative', 'positive'])

# Plotting Confusion Matrix for SVM (Training Set)
svm_train_predictions = svm_classifier.predict(X_train_tfidf)
plot_confusion_matrix(y_train, svm_train_predictions, 'SVM Training Set Confusion Matrix', ['negative', 'positive'])

# Plotting Confusion Matrix for Random Forest (Training Set)
rf_train_predictions = rf_classifier.predict(X_train_tfidf)
plot_confusion_matrix(y_train, rf_train_predictions, 'Random Forest Training Set Confusion Matrix', ['negative', 'positive'])

# Plotting Accuracy Comparison
models = ['SVM Test Set', 'Random Forest Test Set', 'SVM Training Set', 'Random Forest Training Set']
accuracies = [accuracy_score(y_test, svm_predictions), accuracy_score(y_test, rf_predictions),
              accuracy_score(y_train, svm_train_predictions), accuracy_score(y_train, rf_train_predictions)]

plt.figure(figsize=(12, 8))
plt.bar(models, accuracies, color=['green', 'blue', 'orange', 'red'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison - SVM vs Random Forest (Training and Test Sets)')
plt.ylim(0, 1)
plt.show()

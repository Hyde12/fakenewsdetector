import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load labeled dataset from CSV
csv_file_path = 'D:/C backup/Code/School/Twitter Detection/FakeNewsNet.csv'
df = pd.read_csv(csv_file_path)

# Sample data
data = {
    'text': df['title'],
    'label': df['real'],
    'reposts': df['tweet_num']
}

df = pd.DataFrame(data)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels, train_reposts, test_reposts = train_test_split(
    df['text'], 
    df['label'], 
    df['reposts'],
    test_size=0.2, 
    random_state=42
)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(train_data)
tfidf_test = tfidf_vectorizer.transform(test_data)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(tfidf_train, train_labels)

# Make predictions
predictions = classifier.predict(tfidf_test)

# Evaluate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")

# Display classification report
print("Classification Report:")
print(classification_report(test_labels, predictions))

def detect_fake_news_ml(tweet):
    # Transform the tweet using the TF-IDF vectorizer
    tweet_tfidf = tfidf_vectorizer.transform([tweet])

    # Make prediction using the trained classifier
    prediction = classifier.predict(tweet_tfidf)[0]

    return prediction

# Test the fake news detection function
tweet_to_test = input("\n\nPut news that you want to test: ")
result_ml = detect_fake_news_ml(tweet_to_test)
print(f"Result: {result_ml}")
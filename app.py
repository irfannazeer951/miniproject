import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset and train the model
@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf, cv

# Load the model
clf, cv = load_model()

# Streamlit application
st.title("Spam Detection App")
st.write("Enter a message to check if it's spam or ham:")

# Input from user
message = st.text_area("Message")

if st.button("Predict"):
    if message:
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        result = "Spam" if my_prediction[0] == 1 else "Ham"
        st.success(f'The message is: {result}')
    else:
        st.error("Please enter a message.")

if __name__ == '__main__':
    st.run()

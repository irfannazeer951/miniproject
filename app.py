import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset and train the model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    
    # Extract features with CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train the Naive Bayes Classifier
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
        # Transform the input message to the format expected by the model
        data = [message]
        vect = cv.transform(data).toarray()
        
        # Make a prediction
        my_prediction = clf.predict(vect)
        
        # Display the result
        result = "Spam" if my_prediction[0] == 1 else "Ham"
        st.success(f'The message is: {result}')
    else:
        st.error("Please enter a message.")

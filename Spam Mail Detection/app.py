import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Streamlit app title
st.title("Spam Mail Detection App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your mail dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Data preprocessing
    data = df.where((pd.notnull(df)), '')
    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1

    X = data['Message']
    Y = data['Category'].astype('int')

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Feature extraction
    feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extractor.fit_transform(X_train)
    X_test_features = feature_extractor.transform(X_test)

    # Model training
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # Accuracy on training data
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    # Accuracy on test data
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

    # Display accuracy
    st.write("Accuracy on Training Data:", accuracy_on_training_data)
    st.write("Accuracy on Test Data:", accuracy_on_test_data)

    # Input for mail prediction
    st.subheader("Test Your Mail")
    input_your_mail = st.text_area("Enter the mail content here:")

    if st.button("Predict"):
        if input_your_mail:
            input_data_features = feature_extractor.transform([input_your_mail])
            prediction = model.predict(input_data_features)

            if prediction[0] == 1:
                st.success("This is a Ham Mail.")
            else:
                st.error("This is a Spam Mail.")
        else:
            st.warning("Please enter mail content to predict.")
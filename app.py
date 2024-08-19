# Commented out IPython magic to ensure Python compatibility.

import pandas as pd
import numpy as np
import streamlit as st
import pickle

#Path to the model file 
filename = "sentimental_analysis.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

#build a simple streamlit app
st.set_page_config(layout="wide")
st.header('Review Predictor App')

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://img.freepik.com/premium-photo/wide-banner-with-many-random-square-hexagons-charcoal-dark-black-color_105589-1820.jpg" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

# Sidebar content
st.sidebar.subheader("Text Classification App")
st.sidebar.text("Enter your text below to get the sentiment prediction.")

# Text input for prediction
input_text = st.text_area("Enter your review here:")

if st.button("Predict"):
    if input_text:
        # Preprocess the text (if necessary; update with your actual preprocessing)
        processed_text = preprocess_text(input_text)

        # Make prediction
        prediction = loaded_model.predict([processed_text])[0]

        # Display the prediction result
        st.write(f"The predicted sentiment is: **{prediction}**")
    else:
        st.write("Please enter some text to predict.")

# Custom thank you message and GIF
st.write('Thank you for using our app!')
st.markdown("![Thank You](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")

# Optional preprocessing function
def preprocess_text(text):
    # Implement your text preprocessing here
    # For example, lowercasing and removing punctuation
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string

    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

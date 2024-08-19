import pandas as pd
import numpy as np
import streamlit as st
import pickle
import string

# Optional preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Define stopwords
    stop_words = set(["the", "and", "a", "of", "to", "in", "is", "it", "you", "for", "on", "with", "as", "was", "that", "this"])  # Add more stopwords as needed
    
    # Tokenize the text (split by whitespace)
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Path to the model file 
filename = "sentimental_analysis.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

# Build a simple Streamlit app
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
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import string

# Optional preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Define stopwords
    stop_words = set(["the", "and", "a", "of", "to", "in", "is", "it", "you", "for", "on", "with", "as", "was", "that", "this"])  # Add more stopwords as needed
    
    # Tokenize the text (split by whitespace)
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Path to the model file 
filename = "sentimental_analysis.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

# Build a simple Streamlit app
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

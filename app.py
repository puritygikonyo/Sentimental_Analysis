# pip install -U streamlit
# streamlit run app.py

import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open('sentimental_analysis.pkl', 'rb'))

# App title
st.title('Sentiment Analysis Model')

# Option to select Single Review or Batch Processing
option = st.selectbox(
    'Choose Input Method',
    ('Single Review', 'Batch Processing (Excel File)')
)

# Single Review Processing
if option == 'Single Review':
    # Text input for single review
    review = st.text_input('Enter your review:')
    submit = st.button('Predict')

    if submit:
        # Perform sentiment analysis on the input review
        prediction = model.predict([review])

        # Display the result
        if prediction[0] == 'positive':
            st.success('Positive Review')
        else:
            st.warning('Negative Review')

# Batch Processing for Excel File
if option == 'Batch Processing (Excel File)':
    uploaded_file = st.file_uploader("Upload an Excel file with reviews", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)

        # Display the first few rows of the uploaded file
        st.write("### Uploaded Data Preview")
        st.write(df.head())

        # Ensure that the file contains a 'review' column
        if 'review' not in df.columns:
            st.error("The uploaded file must contain a 'review' column.")
        else:
            # Process each review through the model
            df['Sentiment'] = df['review'].apply(lambda x: model.predict([x])[0])

            # Display the DataFrame with the new Sentiment column
            st.write("### Sentiment Analysis Results")
            st.write(df.head())

            # Save the result to a new Excel file
            output_file = "output_with_sentiment.xlsx"
            df.to_excel(output_file, index=False)

            # Provide a download button for the updated Excel file
            st.download_button(
                label="Download Excel with Sentiment",
                data=open(output_file, "rb").read(),
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


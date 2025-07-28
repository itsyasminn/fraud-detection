import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")  

st.title("Credit Card Fraud Detection App")
st.write("Upload transaction data to check for fraud:")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        df = df[feature_columns]


        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

    
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        
        df['Prediction'] = predictions
        df['Fraud Probability'] = probabilities

        st.success("Done! Top 5 predictions:")
        st.dataframe(df[['Prediction', 'Fraud Probability']].head())

        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="fraud_predictions.csv")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


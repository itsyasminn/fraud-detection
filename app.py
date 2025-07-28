import streamlit as st
import pandas as pd
import joblib


model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("features_columns.joblib") 
st.title("Credit Card Fraud Detection App")
st.write("Upload your transaction CSV file. The model will predict if any transactions are fraudulent.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        input_df = df[feature_columns]

    
        if 'Amount' in input_df.columns or 'Time' in input_df.columns:
            input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])

        # Predict
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]

        df['Prediction'] = predictions
        df['Fraud Probability'] = probabilities

        st.success("Prediction completed! Top 5 rows shown below.")
        st.dataframe(df[['Prediction', 'Fraud Probability']].head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Predictions", data=csv, file_name="fraud_predictions.csv")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


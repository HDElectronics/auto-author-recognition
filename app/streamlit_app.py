import streamlit as st
import requests

st.title("Auto Author Recognition Platform")

st.write("""
Upload a text from the training databases, choose your model, and get automatic author recognition with model performance metrics.
""")

# File uploader
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")
    text = file_text
    st.info("Text loaded from file. You can edit it below if needed.")
    st.text_area("Paste your text here:", value=text, key="file_text_area")
else:
    text = st.text_area("Paste your text here:")

model = st.selectbox("Select Model", ["LSTM", "Transformer"])

if st.button("Predict Author"):
    if not text.strip():
        st.warning("Please enter a text.")  
    else:
        payload = {
            "text": text,
        }
        with st.spinner("Predicting author..."):
            try:
                if model == "Transformer":
                    response = requests.post(f"http://localhost:8000/predict/{model.lower()}", json=payload)
                elif model == "LSTM":
                    response = requests.post(f"http://localhost:8000/predict/{model.lower()}", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    pred_num = result['predicted_author']
                    st.success(f"Predicted Author: L{int(pred_num)+1}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

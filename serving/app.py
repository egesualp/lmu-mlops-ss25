import os
import requests
import streamlit as st
import pandas as pd

def get_backend_url():
    """Return the static backend URL for the public API."""
    return "https://financial-sentiment-api-687370715419.europe-west3.run.app"


def classify_text(text: str, backend: str):
    """Send text to backend and return prediction + probabilities."""
    predict_url = f"{backend}/predict"
    try:
        response = requests.post(
            predict_url,
            json={"text": text},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Invalid response from backend: {str(e)}")
        return None


def main():
    """Main Streamlit app."""
    backend = get_backend_url()
    if backend is None:
        st.error("❌ Backend URL could not be resolved.")
        return

    st.title("💬 Financial Sentiment Classifier")
    
    # Debug: Show backend URL
    st.info(f"🔗 Backend URL: {backend}")

    text_input = st.text_area("Enter financial news, tweet, or statement:")

    if st.button("Predict Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter some text.")
            return

        result = classify_text(text_input, backend)

        if result and "label" in result and "score" in result:
            prediction = result["label"]
            score = result["score"]

            # Choose face based on sentiment and confidence
            if prediction == "positive":
                if score > 0.85:
                    face = "😁"
                elif score > 0.6:
                    face = "😀"
                else:
                    face = "🙂"
            elif prediction == "neutral":
                face = "😐"
            elif prediction == "negative":
                if score > 0.85:
                    face = "😢"
                elif score > 0.6:
                    face = "🙁"
                else:
                    face = "☹️"
            else:
                face = "❓"

            st.markdown(
                f"<div style='font-size: 80px; text-align: center;'>{face}</div>",
                unsafe_allow_html=True
            )
            st.success(f"**Predicted sentiment:** `{prediction}`\n\n**Confidence:** {score:.2%}")
        else:
            st.error(f"Failed to get prediction from backend. Response: {result}")


if __name__ == "__main__":
    main()

import os
import requests
import streamlit as st
import pandas as pd

def get_backend_url():
    """Get backend URL from environment variable."""
    return os.environ.get("BACKEND", None)


def classify_text(text: str, backend: str):
    """Send text to backend and return prediction + probabilities."""
    predict_url = f"{backend}/predict/"
    try:
        response = requests.post(
            predict_url,
            json={"text": text},
            timeout=10,
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

        if result:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.success(f"**Predicted sentiment:** `{prediction}`")

            df = pd.DataFrame({
                "Sentiment": ["negative", "neutral", "positive"],
                "Probability": probabilities
            })
            st.bar_chart(df.set_index("Sentiment"))
        else:
            st.error("Failed to get prediction from backend.")


if __name__ == "__main__":
    main()

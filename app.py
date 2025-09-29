import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("🧠 Batch Image Analyzer with Ollama")

# Prompt input
prompt = st.text_area("Enter your prompt (e.g. 'Describe what's happening')", "Describe this image.")

# Upload multiple images
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Analyze button
if st.button("Analyze Images") and uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",  # or llava:7b or other vision-capable model
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
        )

        st.image(image, caption=uploaded_file.name)
        if response.ok:
            result = response.json()
            st.success(result["response"])
        else:
            st.error("Failed to get response from Ollama.")

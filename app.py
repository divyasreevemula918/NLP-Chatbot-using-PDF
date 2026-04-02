import streamlit as st
import os
from src.data_loader import load_pdf_text

uploaded_file = st.file_uploader("Upload PDF", type=["pdf", "txt"])

if uploaded_file is not None:

    # Save file locally
    file_path = os.path.join("temp.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    text = load_pdf_text(file_path)

    st.write("Extracted characters:", len(text))

    if len(text.strip()) == 0:
        st.error("❌ Could not extract text")
        st.stop()

    st.success("✅ File processed successfully")

    # Store text in session
    st.session_state["text"] = text
    query = st.text_input("Ask a question")

if query:
    if "text" not in st.session_state:
        st.error("⚠️ Please upload and process file first")
    else:
        st.write("Processing question...")
        # your LLM logic here
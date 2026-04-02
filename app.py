import streamlit as st
import os
from src.data_loader import load_pdf_text, load_text_file

st.title("NLP Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf", "txt"])

text = ""

if uploaded_file is not None:
    file_name = uploaded_file.name

    with open(file_name, "wb") as f:
        f.write(uploaded_file.read())

    if file_name.endswith(".pdf"):
        text = load_pdf_text(file_name)
    else:
        text = load_text_file(file_name)

    if len(text.strip()) == 0:
        st.error("Could not extract text from the file.")
    else:
        st.success("File uploaded and processed successfully ✅")
        st.write("Extracted characters:", len(text))

# ✅ define query first
query = st.text_input("Ask a question")

if query:
    if len(text.strip()) == 0:
        st.error("Please upload and process a valid file first.")
    else:
        st.write("Your question:", query)
        st.write("Now you can connect this to your vector store / Gemini.")
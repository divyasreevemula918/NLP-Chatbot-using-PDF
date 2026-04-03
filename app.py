import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = load_embedding_model()
    return FAISS.from_texts(chunks, embedding=embeddings)


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def keyword_score(question, sentence):
    q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
    s_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()))

    common = q_words.intersection(s_words)
    score = len(common)

    # boost exact important phrases
    important_phrases = [
        "logistic regression",
        "naive bayes",
        "bayes rule",
        "conditional independence",
        "regularization",
        "gaussian naive bayes",
        "logistic function"
    ]

    q_lower = question.lower()
    s_lower = sentence.lower()

    for phrase in important_phrases:
        if phrase in q_lower and phrase in s_lower:
            score += 5

    return score


def extract_best_answer(question, retrieved_chunks):
    candidate_sentences = []

    for chunk in retrieved_chunks:
        sentences = split_into_sentences(chunk.page_content)
        candidate_sentences.extend(sentences)

    if not candidate_sentences:
        return "The answer is not available in the uploaded PDF."

    scored = []
    for sent in candidate_sentences:
        score = keyword_score(question, sent)
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_sentences = [sent for score, sent in scored[:3] if score > 0]

    if not best_sentences:
        return "The answer is not available in the uploaded PDF."

    answer = " ".join(best_sentences[:2]).strip()

    # avoid formula-only / too short outputs
    if len(answer) < 25:
        return "The answer is not available in the uploaded PDF."

    return answer


def answer_question(question):
    vector_store = st.session_state.vector_store

    docs = vector_store.similarity_search(question, k=4)

    answer = extract_best_answer(question, docs)
    return answer


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading and processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            chunks = split_text_into_chunks(pdf_text)
            st.session_state.chunks = chunks
            st.session_state.vector_store = create_vector_store(chunks)

            st.success("File uploaded and processed successfully ✅")
            st.write(f"Extracted characters: {len(pdf_text)}")
        else:
            st.error("Could not extract text from this PDF.")

if st.session_state.vector_store is not None:
    user_question = st.text_input("Ask a question from the uploaded PDF")

    if user_question:
        with st.spinner("Finding answer..."):
            answer = answer_question(user_question)

        st.subheader("Answer")
        st.write(answer)
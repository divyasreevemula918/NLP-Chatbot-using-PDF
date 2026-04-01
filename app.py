import os
import html
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.data_loader import load_pdf_text, load_text_file
from src.text_splitter import split_text_into_chunks
from src.vector_store import create_vector_store

load_dotenv()

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Session State --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if "raw_text_length" not in st.session_state:
    st.session_state.raw_text_length = 0

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 950px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1e293b;
    }

    .app-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 1.25rem;
    }

    .status-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 18px;
    }

    .status-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .status-text {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.5;
    }

    .chat-heading {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    .message-row-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 14px;
    }

    .message-row-bot {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 14px;
    }

    .message-wrap {
        display: flex;
        align-items: flex-end;
        gap: 10px;
        max-width: 85%;
    }

    .avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }

    .avatar-user {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
    }

    .avatar-bot {
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white;
    }

    .bubble {
        padding: 14px 16px;
        border-radius: 18px;
        font-size: 15.5px;
        line-height: 1.6;
        word-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
    }

    .bubble-user {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff;
        border-bottom-right-radius: 6px;
    }

    .bubble-bot {
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-bottom-left-radius: 6px;
    }

    .label {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 5px;
        padding-left: 4px;
        padding-right: 4px;
    }

    .empty-chat {
        background: #111827;
        border: 1px dashed #334155;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 10px;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.6rem;
    }

    .sidebar-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .sidebar-small {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .metric-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 12px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 NLP Chatbot</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-small">
            Upload a <b>TXT</b> or <b>PDF</b> and chat with your document like ChatGPT 💬
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    file_for_sidebar = st.session_state.current_file_name or "No file selected"
    status_for_sidebar = "Processed ✅" if st.session_state.file_processed else "Waiting ⏳"

    st.markdown(f"""
    <div class="metric-box">
        <div class="sidebar-small"><b>Current File:</b><br>{html.escape(file_for_sidebar)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <div class="sidebar-small"><b>Status:</b><br>{status_for_sidebar}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <div class="sidebar-small"><b>Messages:</b><br>{len(st.session_state.chat_history)}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Main Header --------------------
st.markdown('<div class="app-title">🤖 NLP Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a TXT or PDF file and ask questions from it like ChatGPT 💬</div>',
    unsafe_allow_html=True
)

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Upload your TXT or PDF file", type=["txt", "pdf"])

if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.file_processed = False
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.raw_text_length = 0

    if not st.session_state.file_processed:
        file_name = uploaded_file.name
        temp_path = os.path.join("data", file_name)

        os.makedirs("data", exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing file... ⏳"):
            if file_name.lower().endswith(".pdf"):
                raw_text = load_pdf_text(temp_path)
            else:
                raw_text = load_text_file(temp_path)

            if raw_text and raw_text.strip():
                chunks = split_text_into_chunks(raw_text)
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.file_processed = True
                st.session_state.raw_text_length = len(raw_text)
                st.success(f"File processed successfully: {file_name} ✅")
            else:
                st.error("Could not extract text from the file.")

if uploaded_file is None:
    st.session_state.file_processed = False
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.current_file_name = None
    st.session_state.raw_text_length = 0

# -------------------- Status Card --------------------
current_file = st.session_state.current_file_name or "No file uploaded"
processing_status = "Ready to chat ✅" if st.session_state.file_processed else "Please upload and process a file ⏳"

st.markdown(f"""
<div class="status-card">
    <div class="status-title">📄 Document Status</div>
    <div class="status-text">
        <b>File:</b> {html.escape(current_file)}<br>
        <b>Status:</b> {processing_status}<br>
        <b>Extracted characters:</b> {st.session_state.raw_text_length}
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Chat History --------------------
st.markdown('<div class="chat-heading">💬 Chat</div>', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-chat">
        Upload a file and ask something like:<br><br>
        <b>“What is PCA?”</b><br>
        <b>“Summarize this document.”</b><br>
        <b>“Explain DBSCAN in simple terms.”</b>
    </div>
    """, unsafe_allow_html=True)

for item in st.session_state.chat_history:
    role = item["role"]
    message = html.escape(item["message"])

    if role == "user":
        st.markdown(f"""
        <div class="message-row-user">
            <div class="message-wrap">
                <div>
                    <div class="label" style="text-align:right;">You</div>
                    <div class="bubble bubble-user">{message}</div>
                </div>
                <div class="avatar avatar-user">🧑</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-row-bot">
            <div class="message-wrap">
                <div class="avatar avatar-bot">🤖</div>
                <div>
                    <div class="label">Bot</div>
                    <div class="bubble bubble-bot">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Chat Input --------------------
user_question = st.chat_input("Type your question here...")

if user_question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a TXT or PDF file first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_question
        })

        docs = st.session_state.vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join(docs)

        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = f"""
You are a helpful AI assistant.
Answer the user's question only from the context below.
If the answer is not present in the context, say:
"I could not find that in the uploaded document."

Context:
{context}

Question:
{user_question}
"""

        with st.spinner("Thinking... 🤔"):
            response = llm.invoke(prompt)

        st.session_state.chat_history.append({
            "role": "bot",
            "message": response.content
        })

        st.rerun()
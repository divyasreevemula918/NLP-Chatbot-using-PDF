from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 🔑 Initialize Gemini model (FIXED MODEL NAME)
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",   # ✅ FIXED HERE
        temperature=0.3
    )
    return llm


# 🔗 Create QA Chain
def get_qa_chain(vector_store):
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    return qa_chain


# 💬 Ask Question
def ask_question(qa_chain, query):
    response = qa_chain.invoke({"query": query})

    answer = response["result"]
    sources = response.get("source_documents", [])

    return answer, sources
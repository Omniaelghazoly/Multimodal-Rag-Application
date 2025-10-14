import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
import tempfile
import base64
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# --- Helper Functions ---
def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            base64.b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join([t.page_content for t in docs_by_type["texts"]]) if docs_by_type["texts"] else ""
    prompt_template = f"""
    Answer the question based only on the following context,
    which may include text, tables, and the below images.
    Context: {context_text}
    Question: {user_question}
    """
    prompt_content = [{"type": "text", "text": prompt_template}]
    if docs_by_type["images"]:
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Q&A with Gemini", layout="wide")
st.title("🤖 Multimodal PDF Q&A (Gemini + LangChain)")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.info("💡 Vectorstore created successfully — ready for queries!")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        chain = (
            {"context": retriever | RunnableLambda(parse_docs),
             "question": RunnablePassthrough()}
            | RunnableLambda(build_prompt)
            | ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            | StrOutputParser()
        )

        with st.spinner("Thinking... 🤔"):
            response = chain.invoke({"question": question})

        st.subheader("🔹 Gemini Answer:")
        st.write(response)

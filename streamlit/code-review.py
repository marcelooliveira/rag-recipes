import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

# Step 1: Load and preprocess the Markdown file
file_path = "guia-codificacao.md"
if not os.path.exists(file_path):
    st.error(f"The file {file_path} does not exist.")
    st.stop()

# Load file content into LangChain
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# Adjust chunking for tables with many rows
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increase chunk size to accommodate multiple rows
    chunk_overlap=100,  # Add overlap to preserve table context
    separators=["\n\n", "\n", " "],  # Prioritize splitting by lines
)
split_documents = text_splitter.split_documents(documents)

# Step 2: Clean up Vectorstore
vectorstore_path = "faiss_index"  # Define path for vectorstore storage

# Remove any existing vectorstore files to ensure a fresh start
if os.path.exists(vectorstore_path):
    try:
        for file in os.listdir(vectorstore_path):
            os.remove(os.path.join(vectorstore_path, file))
        os.rmdir(vectorstore_path)
        st.write("Previous vectorstore cleaned successfully.")
    except Exception as cleanup_error:
        st.error(f"Error during vectorstore cleanup: {cleanup_error}")
else:
    st.write("No existing vectorstore to clean.")

# Step 3: Create FAISS Vectorstore and Index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Smaller embedding model
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Step 3: Define a custom prompt for short answers
short_answer_prompt = PromptTemplate(
    input_variables=["context", "code"],
    template=(
        "Você é um especialista em análise de código. Use o guia de codificação a seguir para revisar o código fornecido.\n\n"
        "Guia de Codificação:\n{context}\n\n"
        "Código:\n{question}\n\n"
        "Por favor, liste as violações de regras identificadas no código da seguinte forma:\n"
        "### Regras violadas:\n"
        "Regra N - motivo\n"
        "Regra N - motivo\n"
        "Regra N - motivo\n"
        "Certifique-se de fornecer uma explicação clara para cada violação identificada."
    ),
)

# Set up Retrieval-Augmented Generation (RAG) with the custom prompt
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Increase k for large tables
llm = OllamaLLM(model="phi3", temperature=0.4)  # Deterministic and precise
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": short_answer_prompt},
)

# Step 4: Streamlit Interface
st.title("Code Review Assistant")
st.write("Insira o código abaixo para receber um code review com base no guia de codificação fornecido.")

# Streamlit form for user input
with st.form(key="query_form"):
    user_query = st.text_area("Enter your question:", height=300)
    submit_button = st.form_submit_button(label="Submit")

# Process user query
if submit_button:
    if user_query.strip():
        try:
            response = qa_chain.invoke({"query": user_query})
            answer = response["result"]
            sources = response["source_documents"]
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Sources:")
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown Source')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid question.")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

# Step 1: Load and preprocess the Markdown file
file_path = "fighters.md"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load file content into LangChain
loader = TextLoader(file_path)
documents = loader.load()

# Adjust chunking for tables with many rows
# Use smaller chunks with line-by-line splitting for large tables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,  # Increase chunk size to accommodate multiple rows
    chunk_overlap=100,  # Add overlap to preserve table context
    separators=["\n\n", "\n", " "],  # Prioritize splitting by lines
)
split_documents = text_splitter.split_documents(documents)

# Step 2: Create FAISS Vectorstore and Index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Smaller embedding model
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Step 3: Define a custom prompt for short answers
short_answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert in military aircraft data containing 6 columns: 'Fighter', 'Year', 'Weight', 'Speed', 'Manufacturer', 'Active Units'. Using the following structured information, "
        "answer the user's question specifically, concisely, and accurately:\n\n"
        "Table Data:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# Set up Retrieval-Augmented Generation (RAG) with the custom prompt
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Increase k for large tables
llm = OllamaLLM(model="phi3", temperature=0.0)  # Deterministic and precise
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": short_answer_prompt},
)

# Step 4: Interact with the user
def main():
    print("Stock Query Assistant")
    print("Ask any question about the stock stats in 'stock.md' or type 'exit' to quit.")
    while True:
        query = input("\nYour Question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = qa_chain.invoke({"query": query})
            answer = response["result"]
            sources = response["source_documents"]
            
            print("\nAnswer:")
            print(answer)
            
            print("\nSources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'Unknown Source')}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


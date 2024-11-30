from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import Ollama
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

# Split the documents into smaller chunks for better embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "])
split_documents = text_splitter.split_documents(documents)

# Step 2: Create FAISS Vectorstore and Index
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")  # Smaller embedding model
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Step 3: Define a custom prompt for short answers
short_answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a concise assistant. Based on the following context, "
        "answer the user's question with 2-3 words only.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# Set up Retrieval-Augmented Generation (RAG) with the custom prompt
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = OllamaLLM(model="phi3", temperature=0.0)  # Deterministic and precise
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": short_answer_prompt},
)

# Step 4: Interact with the user
def ask_question(qa_chain):
    """Allows the user to ask a single question."""
    query = input("\nYour Question: ")
    if query.lower() == "exit":
        print("Goodbye!")
        return False
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
    return True

def main():
    print("Query Assistant")
    print("Ask any question about the table or type 'exit' to quit.")
    while True:
        continue_session = ask_question(qa_chain)
        if not continue_session:
            break

if __name__ == "__main__":
    main()

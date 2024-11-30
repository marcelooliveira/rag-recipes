from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Step 1: Load and preprocess the Markdown file
file_path = "stock.md"
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

# Step 3: Set up Retrieval-Augmented Generation (RAG)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = Ollama(model="phi3", temperature=0.0)  # Lower temperature for precise results
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Step 4: Interact with the user
def main():
    print("FIFA Championship Stats Query Assistant")
    print("Ask any question about the FIFA stats in 'fifacup.md' or type 'exit' to quit.")
    while True:
        query = input("\nYour Question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = qa_chain({"query": query})
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

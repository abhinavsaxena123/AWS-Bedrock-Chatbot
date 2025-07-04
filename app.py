import boto3
import streamlit as st
import os

# LangChain Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration
PROMPT_TEMPLATE = """
Human: Use the following pieces of context to provide a concise and detailed
answer to the question. Summarize with at least 250 words, explaining thoroughly.
You can use bullet points if it helps organize the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""

# Directory of PDF documents are stored
PDF_DATA_DIRECTORY = "Data"
# FAISS index save path
FAISS_INDEX_PATH = "faiss_index"

# --- Bedrock Client Initialization ---
@st.cache_resource # Cache the Bedrock client to avoid re-initializing on every rerun
def get_bedrock_client():
    """Initializes and returns a cached Bedrock Runtime client."""
    return boto3.client(
        service_name = "bedrock-runtime",
        region_name = "ap-south-1"
    )

bedrock_client = get_bedrock_client()


# --- Embedding Model Initialization ---
@st.cache_resource
def get_bedrock_embedding_model():
    """Initializes and returns a cached Bedrock Embeddings model."""
    return BedrockEmbeddings(
        model_id = "amazon.titan-embed-text-v2:0",
        client = bedrock_client
    )

bedrock_embedding = get_bedrock_embedding_model()


# --- Document Processing Functions ---
def load_and_split_documents(directory_path: str):
    """
    Loads PDF documents from a specified directory and splits them into chunks.

    Args:
        directory_path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of Document objects, representing chunks of the PDFs.
    """
    st.info(f"Loading documents from: {directory_path}...") 
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()

    # Define text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Size of each text chunk
        chunk_overlap=100,     # Overlap between chunks to maintain context
        length_function=len    # Function to calculate chunk length
    )

    docs = text_splitter.split_documents(documents)
    return docs


def create_and_save_vector_store(documents: list, embedding_model, save_path: str):
    """
    Creates a FAISS vector store from document chunks and saves it locally.

    Args:
        documents (list): A list of document chunks.
        embedding_model: The Bedrock Embeddings model to use.
        save_path (str): The directory path to save the FAISS index.
    """
    vector_store_faiss = FAISS.from_documents(
        documents,
        embedding_model
    )
    vector_store_faiss.save_local(save_path)


def load_vector_store(save_path: str, embedding_model):
    """
    Loads a FAISS vector store from a local directory.

    Args:
        save_path (str): The directory path where the FAISS index is saved.
        embedding_model: The Bedrock Embeddings model used to create the index.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    # allow_dangerous_deserialization=True is required for loading FAISS indexes
    # from disk, as it involves deserializing pickled objects.
    vector_store = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store


# --- LLM Initialization ---
@st.cache_resource # Cache the LLM to avoid re-initializing on every rerun
def get_llm():
    """Initializes and returns a cached Bedrock Chat model (Claude 3 Sonnet)."""
    return BedrockChat(
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
        client = bedrock_client,
        model_kwargs = {
            "temperature": 0.5,
            "max_tokens": 500
        }
    )

# --- RAG Chain Setup ---
def get_rag_chain(llm, vectorstore_faiss):
    """
    Sets up and returns a LangChain RetrievalQA chain.

    Args:
        llm: The initialized Bedrock Chat LLM.
        vectorstore_faiss: The loaded FAISS vector store.

    Returns:
        RetrievalQA: The configured RAG chain.
    """
    rag_prompt = PromptTemplate(
        template = PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Initialize the RetrievalQA chain
    # chain_type="stuff" combines all retrieved documents into one prompt
    # retriever specifies how to fetch relevant documents from the vector store
    # return_source_documents=True to show which documents were used
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type="similarity", # Using similarity search
            search_kwargs={
                "k":5    # Retrieve top 5 most similar documents
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    return qa_chain


# Streamlit Application
def main():
    st.set_page_config(page_title="RAG: Bedrock Chatbot for Cloud Computing Research", layout="wide")
    st.header("End-to-End RAG Chatbot with Amazon Bedrock")

    with st.sidebar:
        st.subheader("Vector Store Management")

        # Explicit button to create/update vector store
        if st.button("Create/Update Vector Store"):
            if not os.path.exists(PDF_DATA_DIRECTORY) or not os.listdir(PDF_DATA_DIRECTORY):
                st.warning(f"'{PDF_DATA_DIRECTORY}' directory is empty or does not exist. Please place your PDFs inside.")
            else:
                with st.spinner("Processing documents and creating vector store..."):
                    docs = load_and_split_documents(PDF_DATA_DIRECTORY)
                    create_and_save_vector_store(docs, bedrock_embedding, FAISS_INDEX_PATH)
                    st.success("Vector Store updated successfully!")
        
        st.markdown("---") 

    # Main area for chatbot interaction
    # Initial one-time setup if FAISS index doesn't exist
    if not os.path.exists(FAISS_INDEX_PATH):
        if not os.path.exists(PDF_DATA_DIRECTORY) or not os.listdir(PDF_DATA_DIRECTORY):
            st.warning(f"'{PDF_DATA_DIRECTORY}' directory is empty or does not exist. Please place your PDFs inside and click 'Create/Update Vector Store' in the sidebar.")
            st.stop() 
        else:
            with st.spinner("One-time setup: Processing documents and creating vector store... This may take a moment."):
                docs = load_and_split_documents(PDF_DATA_DIRECTORY)
                create_and_save_vector_store(docs, bedrock_embedding, FAISS_INDEX_PATH)
            st.success("Initial Vector Store created successfully! You can now ask questions.")


    st.markdown("---")
    st.subheader("Ask a question about the uploaded research papers on Cloud Computing!") 

    user_question = st.text_area("Your Question:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Fetching answer..."):
                try:
                    # Load the FAISS index
                    vectorstore_faiss = load_vector_store(FAISS_INDEX_PATH, bedrock_embedding)

                    # Get the LLM
                    llm = get_llm()

                    # Get the RAG chain
                    qa_chain = get_rag_chain(llm, vectorstore_faiss)

                    # Get the response
                    answer = qa_chain({"query": user_question})

                    st.subheader("Answer:") 
                    st.write(answer['result'])

                    # display source documents
                    with st.expander("🔗 See Source Documents"): 
                        for i, doc in enumerate(answer['source_documents']):
                            st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}")
                            st.write(doc.page_content)
                            st.markdown("---")

                except Exception as e:
                    st.error(f"An error occurred: {e}. Please ensure your AWS credentials are set up correctly and the Bedrock region/model are accessible.")
        else:
            st.warning("Please enter a question to get an answer!")


# --- Entry Point ---
if __name__ == "__main__":
    main()

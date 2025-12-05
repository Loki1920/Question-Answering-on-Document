import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
import tempfile

# --- Setup and Configuration ---

# Load environment variables from a .env file for secure API key management
load_dotenv()
# api key from env
groq_api_key = os.getenv("GROQ_API_KEY")

# Set the title for the Streamlit web application
st.title("Document Q&A 🤖")
st.markdown("Upload a PDF, and I'll answer your questions about it!")


# --- Backend RAG Pipeline Logic ---

@st.cache_resource
def create_vector_store(file_path):
    """
    Creates a FAISS vector store from a given PDF file path.

    This function is cached to prevent re-processing the same document,
    improving performance significantly on subsequent runs.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        FAISS: A FAISS vector store object containing the document embeddings,
               or None if an error occurs.
    """
    if file_path:
        try:
            # 1. Load the document
            st.write("Loading PDF...")
            pdf_loader = PyPDFLoader(file_path)
            documents = pdf_loader.load()

            # 2. Split the document into chunks for processing
            st.write("Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # 3. Create embeddings using a fast, local model
            st.write("Creating embeddings...")
            embeddings = FastEmbedEmbeddings()

            # 4. Create the FAISS vector store from the chunks and embeddings
            st.write("Creating vector store...")
            vector_store = FAISS.from_documents(docs, embeddings)
            st.success("Vector store created successfully!")
            return vector_store

        except Exception as e:
            st.error(f"Error processing the PDF file: {e}")
            return None
    return None

# --- Frontend Streamlit UI ---

# Initialize session state variables to persist data across reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for file uploading and processing
with st.sidebar:
    st.header("Upload your PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if pdf_file:
        # On button click, process the document
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                # Create a temporary file to get a stable path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Create and store the vector store in the session state
                st.session_state.vector_store = create_vector_store(tmp_file_path)

                # Clean up the temporary file
                os.remove(tmp_file_path)

# Main chat interface for asking questions
st.header("Ask a Question")
prompt = st.text_input("Ask a question about the document:")

# Display a warning if the document hasn't been processed yet
if not st.session_state.vector_store and prompt:
    st.warning("Please upload and process a document first.")

# Handle the Q&A logic if a document is processed and a question is asked
if st.session_state.vector_store and prompt:
    with st.spinner("Searching for the answer..."):
        try:
            # Initialize the Groq LLM with the high-speed Llama3 model
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

            # Define the prompt template to guide the LLM's responses
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are an intelligent assistant for question-answering tasks.
                Answer the user's questions based on the provided context only.
                If the answer is not available in the context, clearly state that you don't know.
                Provide a concise, accurate, and helpful answer.

                <context>
                {context}
                </context>

                Question: {input}
                """
            )

            # Create the document chain that combines the LLM and the prompt
            document_chain = create_stuff_documents_chain(llm, prompt_template)

            # Create the retriever from the vector store to fetch relevant documents
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

            # Create the full retrieval chain that connects the retriever and the document chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Invoke the chain with the user's question to get a response
            response = retrieval_chain.invoke({"input": prompt})

            # Display the final answer
            st.write("### Answer")
            st.write(response["answer"])

            # BONUS: Display source documents for attribution and transparency
            with st.expander("Show Document Sources"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Source {i+1}:** (Page: {doc.metadata.get('page', 'N/A')})")
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")


Groq-Powered Document Q&A Chatbot 🤖
This project is a powerful, yet simple, document question-answering application built with Streamlit. It leverages the high-speed Groq LPU for inference, LangChain for its RAG (Retrieval-Augmented Generation) pipeline, and FAISS for efficient in-memory vector storage.

Users can upload a PDF document, and the application will create a searchable vector store. Subsequently, users can ask questions in natural language, and the application will provide contextually-aware answers based only on the content of the uploaded document.

✨ Features
Intuitive UI: A clean and simple user interface powered by Streamlit.

PDF Document Support: Upload any PDF file to begin the Q&A session.

High-Speed Inference: Utilizes the Groq API with the Llama3-8b model for near-instantaneous answer generation.

Advanced RAG Pipeline: Implements a robust RAG workflow using LangChain to ensure answers are grounded in the document's context, minimizing hallucinations.

Local & Fast Embeddings: Uses FastEmbedEmbeddings to quickly generate vector representations of document chunks without needing a separate API call.

Efficient In-Memory Search: Employs FAISS (Facebook AI Similarity Search) for rapid semantic searching of the most relevant document chunks.

Source Attribution: For transparency, the application displays the exact source chunks from the document that were used to generate the answer.

Smart Caching: Streamlit's resource caching is used to avoid reprocessing the same document, making the experience much faster on subsequent interactions.

🛠️ Tech Stack
Frontend: Streamlit

LLM API: Groq (Llama3-8b-8192)

Core Framework: LangChain

Vector Store: FAISS (faiss-cpu)

Embeddings: FastEmbed

Document Loading: PyPDF

Environment Management: python-dotenv

🚀 Getting Started
Follow these instructions to set up and run the project locally.

1. Prerequisites
Python 3.8+

A GroqCloud API Key. You can get one for free at GroqCloud Console.

2. Setup and Installation
Step 1: Clone the repository (or create the project directory)

If you are cloning a repository:

git clone <your-repo-url>
cd <your-repo-name>

Alternatively, just ensure your files are in a dedicated project folder.

Step 2: Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

Step 3: Create a .env file

Create a file named .env in the root of your project directory and add your Groq API key:

GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Replace the placeholder with your actual Groq API key.

Step 4: Install Dependencies

Create a requirements.txt file with the following content:

streamlit
langchain
langchain-groq
python-dotenv
pypdf
faiss-cpu
langchain_community
fastembed

Now, install all the required libraries using pip:

pip install -r requirements.txt

3. Running the Application
Once the setup is complete, you can run the Streamlit application with a single command:

streamlit run app_1.py

Your web browser will automatically open a new tab with the running application.

⚙️ How It Works
The application follows a standard Retrieval-Augmented Generation (RAG) architecture:

Upload & Process: The user uploads a PDF via the Streamlit interface. The application uses PyPDFLoader to load the document's content.

Chunking: The loaded text is split into smaller, manageable chunks using RecursiveCharacterTextSplitter. This ensures that the context provided to the LLM is focused.

Embedding & Storage: Each text chunk is converted into a numerical vector (embedding) using FastEmbedEmbeddings. These embeddings are then stored in a FAISS vector store, which creates a searchable index in memory.

Question & Retrieval: When the user asks a question, the application creates an embedding for the question and uses the FAISS index to perform a semantic search, retrieving the most relevant text chunks from the document.

Generation: The retrieved chunks (the context) and the user's original question are passed to the Groq Llama3 model through a carefully crafted prompt. The LLM generates a final answer that is based solely on the provided context.

Display: The answer and its source chunks are displayed back to the user in the Streamlit UI.

Enjoy your intelligent document assistant! cheers

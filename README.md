# Fly-Kite Airline RAG App

A modular Retrieval-Augmented Generation (RAG) application for Fly-Kite Airlines, designed to deliver accurate, context-aware responses using custom document embeddings and chunking strategies.

## 🧠 How It Works

1. **Chunking**: Splits airline documents into manageable chunks.
2. **Embedding**: Converts chunks into vector representations.
3. **Retrieval + Generation**: Uses similarity search to retrieve relevant chunks and generate responses.


setup the .env file with below key and values

GROQ_API_KEY=<<groq key>>

OPEN_API_KEY=<<open api key>>

HF_TOKEN=<<HF TOKEN>>

CHUNK_SIZE=500

CHUNK_OVERLAP=20

SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2

GROQ_LLM_MODEL=llama-3.3-70b-versatile

CHROMA_DB_PATH=chroma_db

PDF_FILE_NAME=PDF_FILES/Flykite Airlines_ HRP.pdf



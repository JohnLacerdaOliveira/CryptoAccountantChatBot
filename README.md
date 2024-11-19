# CryptoAccountantChatBot

## Overview

This project is a **document-based question-answering system** built using LangChain, OpenAI, FAISS, and Gradio. It processes and embeds PDF documents into a vector database, allowing users to query documents through natural language. Relevant context is retrieved from the database and used to generate answers via a language model.

## Features

- **PDF Loading**: Reads PDF documents for processing.
- **Text Splitting**: Splits large documents into manageable chunks with overlapping for better embedding and retrieval.
- **Embeddings**: Utilizes OpenAI's embeddings for vectorizing document content.
- **Vector Search**: Implements similarity search using FAISS for retrieving the most relevant content.
- **Context-Aware Q&A**: Builds detailed responses based on retrieved document context.
- **Interactive Interface**: Provides a user-friendly chat interface powered by Gradio.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see [Installation](#installation))

## Installation

## Installation

1. **Clone the Repository**  
   Run the following commands to clone the repository and navigate into the project directory:

   ```bash
   git clone https://github.com/JohnLacerdaOliveira/CryptoAccountantChatBot.git

   ```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Set Environment Variables Create a .env file in the root directory and add your OpenAI API key:**

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   EMBEDDING_MODEL=text-embedding-model-name
   LLM_MODEL=llm-model-name

   ```

##File Structure
├── Documents/ # Folder to store PDF documents
├── main.py # Main application script
├── .env # Environment variables
├── requirements.txt # List of dependencies
└── README.md # Project documentation

## Key Functions

### `document_loader(path: str)`

Loads PDF files and extracts their content.

### `doc_splitter(docs)`

Splits documents into chunks for better embedding performance.

### `embed_docs(splitted_docs)`

Embeds the text chunks using OpenAI embeddings and stores them in a FAISS vector database.

### `similarity_search(user_prompt: str)`

Searches the vector database for the most relevant document chunks.

### `create_system_prompt(context: str, user_prompt: str)`

Generates a prompt for the language model using retrieved document context.

### `predict(user_question, history)`

Handles user queries by retrieving context and generating responses.

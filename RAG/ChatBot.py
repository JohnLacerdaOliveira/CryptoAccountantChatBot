# Import necessary libraries
import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
DOCUMENTS_PATH = os.getcwd() + "\Documents"
SINGLE_DOCUMENT = os.getcwd() + "\Documents\Edited_A tributação dos Criptoativos.pdf"

# Initialize global vector database
vector_db = None

def document_loader(path: str):
  #loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.pdf", loader_cls=PyPDFLoader, use_multithreading = True)
  loader = PyPDFLoader(path, extract_images = False)
  docs = loader.load()
  return docs

def doc_splitter(docs):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
  splitted_docs = text_splitter.split_documents(docs)
  return splitted_docs

def embed_docs(splitted_docs):
  embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL) 
  vector_db = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
  return vector_db

def process_documents():
  global vector_db  
  
  try:  
    docs = document_loader(SINGLE_DOCUMENT)
    splitted_docs = doc_splitter(docs)
    vector_db = embed_docs(splitted_docs)

    if vector_db is None:
      raise RuntimeError("The database has failed to initialize.")

  except Exception as e:
    print(f"Error initializing vector database: {e}")

process_documents();


def similarity_search(user_prompt:str):
  return vector_db.similarity_search(user_prompt, k=5)

def create_system_prompt(context:str, user_prompt:str):
  return f"""
      You are an assistant for question-answering tasks.
      Use only the following pieces of retrieved context from a document to answer the user question.
      If the document doesn't provide the answer, tell the user you don't know that information.
      
      **Context**
      {context}
      **End of Context**
      **User query**
      {user_prompt}
      **End of user query**
  """

def create_context(relevant_docs):
  content_list = []
    
  # Iterate over each document in the relevant_docs list
  for doc in relevant_docs:
      # Extract the page content from the current document
      page_content = doc.page_content
      
      # Append the extracted page content to the content_list
      content_list.append(page_content)
  
  # Join all the page content strings into a single string with double newlines separating them
  return "\n".join(content_list)
  

def build_chat_history(history):   
  messages = []
  count = 0

  # Check if there is any conversation history
  if len(history) > 0:
      # Iterate through each message in the first element of the history (assuming history is a list of lists)
      for message in history[0]:
          # Print the current message for debugging purposes
          print(message)

          if count % 2 == 0:  # Even count means the message is from the human user
              messages.append({"role": "human", "content": message})  # Append the message with role "human"
          else:  # Odd count means the message is from the assistant
              messages.append({"role": "assistant", "content": message})  # Append the message with role "assistant"

          count += 1
          
  return messages
   
def predict(user_question, history):

  relevant_docs = similarity_search(user_question)
  context = create_context(relevant_docs)

  llm_prompt= create_system_prompt(context, user_question)

  # Prepare the LLM input with conversation history
  llm = ChatOpenAI(model=LLM_MODEL)
  
  previous_messages = build_chat_history(history)

  # After processing the history, append the system prompt to the messages list
  previous_messages.append({"role": "system", "content": llm_prompt})

  # Append the current user query to the messages list as a "human" role message
  previous_messages.append({"role": "human", "content": user_question})

  response = llm.invoke(previous_messages)

  return response.content



# Launch Gradio interface
if __name__ == "__main__":
    gr.ChatInterface(predict).launch(debug=True)

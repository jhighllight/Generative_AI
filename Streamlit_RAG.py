# Import Librariesm
import streamlit as st
import tempfile
import shutil
import os
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Constants and API Keys
OPENAI_API_KEY = "..."
GPT_MODEL_NAME = 'gpt-4'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Function Definitions
def load_and_split_document(uploaded_file):
    if uploaded_file is not None:
        # BytesIO 객체를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # 임시 파일 경로를 사용하여 PyPDFLoader 초기화
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
            st.write(f"Loaded and split {len(pages)} pages.")
            return pages
        finally:
            # 사용 후 임시 파일 삭제
            os.remove(tmp_file_path)
    return []

def process_uploaded_files(uploaded_files):
    all_pages = []
    for uploaded_file in uploaded_files:
        pages = load_and_split_document(uploaded_file)
        all_pages.extend(pages)
    st.write(f"Total pages from all documents: {len(all_pages)}")
    return all_pages

def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks

def create_embeddings(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.write("Embeddings object created.")
    return embeddings

def setup_vector_database(documents, embeddings):
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=None)
    st.write(f"Vector database setup with {len(documents)} documents.")
    return vectordb

def initialize_chat_model(api_key, model_name):
    chat_model = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)
    st.write("Chat model initialized.")
    return chat_model

def create_retrieval_qa_chain(chat_model, vector_database):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=vector_database.as_retriever(), memory=memory)
    st.write("Retrieval QA chain created.")
    return qa_chain

def ask_question_and_get_answer(qa_chain, question):
    try:
        response = qa_chain({"question": question})
        if not response['answer']:
            st.write("No answer found. Please refine your question.")
        else:
            return response['answer']
    except Exception as e:
        st.error(f"Error processing the question: {e}")
        return "Unable to generate an answer due to an error."

# Main Execution Flow
def main():
    st.title("Medical Research Report Analysis")
    
    uploaded_files = st.file_uploader("Upload Medical Research Reports", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner('Loading and processing the documents...'):
            pages = process_uploaded_files(uploaded_files)
            documents = split_text_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = create_embeddings(OPENAI_API_KEY)
            vector_database = setup_vector_database(documents, embeddings)
            chat_model = initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
            qa_chain = create_retrieval_qa_chain(chat_model, vector_database)
            st.success('Documents processed successfully!')

        question = st.text_input("Enter your question about the Medical Research Reports:")
        if question:
            with st.spinner('Finding the answer...'):
                answer = ask_question_and_get_answer(qa_chain, question)
                if answer:
                    st.write(f"Answer: {answer}")
                else:
                    st.write("No answer could be generated for your question.")

if __name__ == "__main__":
    main()
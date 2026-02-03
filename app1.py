# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 19:52:16 2026

@author: Home
"""

from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings('ignore')
import os

app = Flask(__name__)

# 1. Initialize RAG Components
os.environ["GOOGLE_API_KEY"] = "********Google API Key********"

loader = PyPDFLoader("Pharmacy Dictionary.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# 3. Create Embeddings using Gemini models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector Store (Chroma handles the searching)
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./pharmacy_local_db"
)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)

# 6. Build the Retrieval Prompt
system_prompt = (
    "You are a pharmaceutical expert. Use the dictionary context provided "
    "to explain the term or medication. If the term is not in the context, "
    "inform the user but provide general medical knowledge as a backup.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 7. Create and Run the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400
    
    response = rag_chain.invoke({"input": user_input})
    return jsonify({"answer": response["answer"]})

if __name__ == '__main__':

    app.run(debug=True, port=8080)

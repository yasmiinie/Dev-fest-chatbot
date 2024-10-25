import cohere
import os
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import TextLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.schema import Document

app = Flask(__name__)
CORS(app)

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

doc_url = os.getenv("doc_url")


response = requests.get(doc_url)

if response.status_code == 200:
    document_content = response.text
    print("Document content retrieved successfully.")
else:
    print("Failed to retrieve document content. Status code:", response.status_code)

documents = [Document(page_content=document_content)]

docs = text_splitter.split_documents(documents)

# Initialize embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(docs, embedding_model)

# Initialize Cohere client
cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(cohere_api_key)

def cohere_generate(prompt):
    try:
        response = co.generate(
            model='command-xlarge-nightly',  
            prompt=prompt,
            max_tokens=200
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def search_with_langchain(query):
    # Search in FAISS vector store
    docs = vectorstore.similarity_search(query)
    if docs:
        context = " ".join([doc.page_content for doc in docs])
        prompt = f"""
Your name is B4, and you are a helpful and concise assistant for Users in our plateform Core Capital. You must answer their question . 
Do not add extra details or unrelated information.

Context: {context}
Question: {query}
Answer:
"""

        # Generate response with Cohere
        response = cohere_generate(prompt)
    else:
        response = "No relevant information found."
    return response

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get("message")  # Get the question 
    if query:
        bot_response = search_with_langchain(query)
        return jsonify({"response": str(bot_response)})
    return jsonify({"error": "No message received"}), 400

if __name__ == '__main__':
    app.run(debug=False)
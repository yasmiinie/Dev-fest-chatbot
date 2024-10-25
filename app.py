import cohere
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

doc_url = os.getenv("doc_url")
response = requests.get(doc_url)

# Load document content
if response.status_code == 200:
    document_content = response.text
    print("Document content retrieved successfully.")
else:
    print("Failed to retrieve document content. Status code:", response.status_code)

# Embed document
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
document_embeddings = embedder.encode([document_content])

# Initialize FAISS
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Initialize Cohere
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

def search_with_faiss(query):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=1)  # Fetch closest match
    if len(I) > 0 and D[0][0] < 0.5:
        context = document_content
        prompt = f"""
Your name is B4, and you are a helpful and concise assistant for Users in our platform Core Capital. You must answer their question. 

Context: {context}
Question: {query}
Answer:
"""
        return cohere_generate(prompt)
    return "No relevant information found."

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get("message")
    if query:
        bot_response = search_with_faiss(query)
        return jsonify({"response": str(bot_response)})
    return jsonify({"error": "No message received"}), 400

if __name__ == '__main__':
    app.run(debug=False)

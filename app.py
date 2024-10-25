import cohere
import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

# Load document content from URL
doc_url = os.getenv("doc_url")
response = requests.get(doc_url)
if response.status_code == 200:
    document_content = response.text
else:
    document_content = "Error: Document not found."

# Initialize Cohere client
cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(cohere_api_key)

def generate_embeddings(texts):
    response = co.embed(model="small", texts=texts)
    return response.embeddings

# Embed document content and store
documents = [document_content]
doc_embeddings = generate_embeddings(documents)

def search_and_generate_response(query):
    query_embedding = generate_embeddings([query])[0]
    
    # Basic similarity check to find relevant content
    best_doc = max(doc_embeddings, key=lambda d: sum(a * b for a, b in zip(d, query_embedding)))
    
    prompt = f"Context: {best_doc}\nQuestion: {query}\nAnswer:"
    response = co.generate(model="command-xlarge-nightly", prompt=prompt, max_tokens=200)
    return response.generations[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get("message")
    if query:
        response_text = search_and_generate_response(query)
        return jsonify({"response": response_text})
    return jsonify({"error": "No message received"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)

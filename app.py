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

# Split document into smaller chunks (e.g., paragraphs)
chunk_size = 500  # Adjust as needed
chunks = [document_content[i:i + chunk_size] for i in range(0, len(document_content), chunk_size)]

# Initialize Cohere client
cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(cohere_api_key)

# Generate embeddings for each chunk
def generate_embeddings(texts):
    response = co.embed(model="small", texts=texts)
    return response.embeddings

# Embed document chunks and store
doc_embeddings = generate_embeddings(chunks)

# History for each session (to be cleared/reset as needed)
session_history = {}

# Function to handle the initial question
def initial_question_handler(query, session_id):
    # Directly respond based on document content without requiring prior context
    query_embedding = generate_embeddings([query])[0]
    
    # Find the most relevant chunk
    best_chunk, best_score = None, -1
    for chunk, doc_embedding in zip(chunks, doc_embeddings):
        score = sum(a * b for a, b in zip(doc_embedding, query_embedding))
        if score > best_score:
            best_chunk, best_score = chunk, score

    prompt = f"""
Your name is B4, and you are a helpful assistant for Core Capital. You must answer strictly based on the context provided in the chunk. 
Do not add extra details or unrelated information. 
If you don't know the answer, respond with: "Sorry, I don't know."

Context: {best_chunk}
Question: {query}
Answer:
"""
    
    response = co.generate(model="command-xlarge-nightly", prompt=prompt, max_tokens=200)
    
    # Store initial response and query in session history
    session_history[session_id] = query  # Saving initial query
    return response.generations[0].text.strip()

# Function for follow-up questions
def search_and_generate_response(query, session_id):
    query_embedding = generate_embeddings([query])[0]
    
    # Find the most relevant chunk based on cosine similarity
    best_chunk, best_score = None, -1
    for chunk, doc_embedding in zip(chunks, doc_embeddings):
        score = sum(a * b for a, b in zip(doc_embedding, query_embedding))
        if score > best_score:
            best_chunk, best_score = chunk, score
    
    # Include previous query if it's a follow-up
    previous_query = session_history.get(session_id, "")
    if previous_query:
        combined_query = f"Previous question: {previous_query}\nFollow-up question: {query}"
    else:
        combined_query = query

    # Update the history for next reference
    session_history[session_id] = query

    prompt = f"""
Your name is B4, and you are a helpful assistant for Core Capital. You must answer strictly based on the context provided in the chunk. 
Do not add extra details or unrelated information.If you don't know the answer, respond with: "Sorry, I don't know."

Context: {best_chunk}
Question: {combined_query}
Answer:
"""

    response = co.generate(model="command-xlarge-nightly", prompt=prompt, max_tokens=200)
    return response.generations[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get("message")
    session_id = request.json.get("session_id")  # Unique ID for each session (like user ID)
    if query and session_id:
        # Check if it's the first question
        if session_id not in session_history:
            response_text = initial_question_handler(query, session_id)
        else:
            response_text = search_and_generate_response(query, session_id)
        return jsonify({"response": response_text})
    return jsonify({"error": "No message received or session ID missing"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)

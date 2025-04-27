from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
import requests
# Load environment variables from .env
load_dotenv()

# Flask app init
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
ollama_url = "http://localhost:11434/api/generate"

headers = { 
    "Content-Type": "application/json",
    "Accept": "application/json"
}
data = {
    "model": "llama3",
    "prompt": "",
    "stream": False,
    "stop_sequence": "\n"
}

# Init Pinecone
pc = Pinecone(api_key=os.getenv("pinecapikey"))
index_name = "nlp-research-project"

try:
    index = pc.Index(index_name)
except Exception as e:
    index = None
    print(f"Error initializing Pinecone index: {e}")

# Init Gemini
genai.configure(api_key=os.getenv("gemini_api_key"))
gemini_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction="If the question is not related to the context, say 'I don't know'.",
    generation_config=genai.GenerationConfig(
        max_output_tokens=500,
        top_k=2,
        top_p=0.5,
        temperature=0.5,
        stop_sequences=["\n"]
    )
)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Vector search with Gemini answer
@app.route('/search/<query>', methods=['GET'])
def search(query):
    if not query:
        return "Query parameter is required", 400

    if not index:
        return "Pinecone index is not initialized", 500

    gemini_answer = "No answer found"
    try:
        # Get embedding for the query
        query_vector = model.encode(query).tolist()

        # ➤ Vector search
        vector_results = index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )

        vector_chunks = [{
            "id": match["id"],
            "score": round(match["score"], 2),
            "text": match["metadata"].get("text", "")
        } for match in vector_results.get("matches", [])]

        # Combine the top 5 results for context
        combined_context = "\n".join([chunk["text"] for chunk in vector_chunks[:5]])

        # Generate Gemini answer
        gemini_answer = generate_gemini_response(combined_context, query)
        # Generate Ollama Llama answer
        llama_answer = generate_ollama_llama_response(combined_context, query)
        print(f"LLama answer: {llama_answer}")

    except Exception as e:
        print(f"Error during vector search or Gemini response generation: {e}")
    
    return render_template('vector_search_results.html', query=query, results=vector_chunks, gemini_answer=gemini_answer, llama_answer=llama_answer)

def generate_gemini_response(context, query):
    try:
        # Generate response using Gemini with proper string formatting
        response = gemini_model.generate_content(f"Context: {context}\n\nQuestion: {query}")
        answer = response.text # Assuming the response contains a "content" key
        
        if not answer:
            return "No answer found"
        
        # Clean up the answer (optional based on response format)
        answer = answer.strip()
        
        if answer == "I don't know":
            return "I don't know"
        else:
            return answer
    
    except Exception as e:
        # Add error handling
        return f"Error generating response: {str(e)}"
    
def generate_ollama_llama_response(context, query):
    try:
        # Generate response using Ollama Llama
        data["prompt"] = f"Context: {context}\n\nQuestion: {query}"
        response = requests.post(url=ollama_url,headers=headers, json=data)
        
        if response.status_code == 200:
            answer = response.json().get("response", "")
            print(f"LLama response: {answer}")
            # Clean up the answer (optional based on response format)
            answer = answer.strip()
            if not answer:
                return "No answer found"
            return answer.strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
  
    
    except Exception as e:
        # Add error handling
        return f"Error generating response: {str(e)}"
"abcd"
if __name__ == "__main__":
    app.run(debug=True)

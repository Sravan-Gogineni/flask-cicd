from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
import requests
import subprocess
import time
import psutil  # For checking and terminating processes

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
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
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
    llama_answer = "No answer found"
    ollama_process = None

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

        # Start Ollama server if not already running
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for Ollama server to initialize with a timeout
        timeout = 10  # Set a 10-second timeout for Ollama to start
        for _ in range(timeout):
            if is_ollama_running():
                break
            time.sleep(1)
        else:
            print("Ollama server did not start in time, skipping Llama response generation.")
            return render_template('vector_search_results.html', query=query, results=vector_chunks, gemini_answer=gemini_answer, llama_answer="Ollama server timeout.")

        # Generate Ollama Llama answer
        llama_answer = generate_ollama_llama_response(combined_context, query)

        print(f"Llama answer: {llama_answer}")

    except Exception as e:
        print(f"Error during vector search or Gemini response generation: {e}")

    finally:
        # If Ollama process was started, terminate it after returning the result
        if ollama_process:
            print("Terminating Ollama server...")
            terminate_process(ollama_process)

    return render_template('vector_search_results.html', query=query, results=vector_chunks, gemini_answer=gemini_answer, llama_answer=llama_answer)


# Helper functions for generating responses
def generate_gemini_response(context, query):
    try:
        # Generate response using Gemini with proper string formatting
        response = gemini_model.generate_content(f"Context: {context}\n\nQuestion: {query}")
        answer = response.text  # Assuming the response contains a "content" key
        
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
        response = requests.post(url=ollama_url, headers=headers, json=data)
        
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


# Check if Ollama server is running (using psutil)
def is_ollama_running():
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if 'ollama' in proc.info['name'].lower():
            return True
    return False

# Terminate the Ollama process
def terminate_process(process):
    try:
        process.terminate()
        process.wait(timeout=5)  # Wait for it to terminate cleanly
    except psutil.NoSuchProcess:
        print("Ollama process already terminated.")
    except Exception as e:
        print(f"Error terminating Ollama process: {e}")
        try:
            # Forcefully kill the process if terminate does not work
            process.kill()
        except Exception as kill_error:
            print(f"Error killing Ollama process: {kill_error}")
"vsddddd"

if __name__ == "__main__":
    app.run(debug=True)

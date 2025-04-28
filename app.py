from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
import requests
import subprocess
import time
import psutil  # For checking processes
from rouge_score import rouge_scorer, scoring
import json
import logging
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
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

data1 = {
    "model": "deepseek-r1:1.5b",
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

ground_truth_answer =  {
    "Define external fusion and how it is helpful to LLM": "External fusion methods integrate multi-layer visual features at the input stage before feeding visual tokens into the LLMs.",
    "Different of cohesion phenomena": "Different cohesion phenomena include: 1. Repetition 2. Synonymy 3. Conjunction 4. Ellipsis 4. Substitution 6. Reference.",
    "Issues with modular, rule-based transformations of linguistic phenomena": "The limitations are Limited Coverage: ,Linguistic Diversity, Dependence on Grammar Systems",
    "convergence to B-stationarity?": "is a desirable property, it is often established under restrictive assumptions. In particular, if the MPCC linear independence constraint qualification (MPCC-LICQ) holds (i.e., the equality constraints and active inequality constraints in (12) are linearly independent), then B-stationarity is equivalent to S- stationarity ([27]).",
    "What is TCM?": "TCM is Traditional Chinese Medicine",
    "What is Part of Speech Tagging": "NLP task that involves labeling each word in a sentence with its part-of-speech,",
    "What are stress tests? How are they evaluated": "To evaluate the robustness of the selected models when training with the dataset, four of the strategies for stress test generation from (Naik et al., 2018) are used: * Length mismatch: Make the premise a lot longer than the hypothesis, by adding the expression: y verdadero es verdadero y verdadero es verdadero y verdadero es verdadero y verdadero es verdadero y verdadero es verdadero, which does not alter the premise meaning. * Negation: Add negation to the hypothesis without altering the meaning with the expression: y falso no es verdadero. * Overlap: Add the expression: y verdadero es verdadero to the hypothesis to generate a word mismatch between premise and hypothesis. * Spelling: Misspell a word in the premise."
}


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
    deepseek_answer = "No answer found"
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
        deepseek_answer = generate_ollama_deepseek_response(combined_context, query)
        
        # Evaluate answers using ROUGE and BLEU scores

        gemini_rogue_score = calculate_rouge(query, gemini_answer, ground_truth_answer)
        gemini_bleu_score = calculate_bleu_score(query, gemini_answer, ground_truth_answer)
        llama_rogue_score = calculate_rouge(query, llama_answer, ground_truth_answer)
        llama_bleu_score = calculate_bleu_score(query, llama_answer, ground_truth_answer)
        deepseek_rogue_score = calculate_rouge(query, deepseek_answer, ground_truth_answer)
        deepseek_bleu_score = calculate_bleu_score(query, deepseek_answer, ground_truth_answer)

    except Exception as e:
        print(f"Error during vector search or Gemini response generation: {e}")

    return render_template('vector_search_results.html', query=query, results=vector_chunks, gemini_answer=gemini_answer, llama_answer=llama_answer , deepseek_answer=deepseek_answer,
                           gemini_rogue_score=gemini_rogue_score, gemini_bleu_score=gemini_bleu_score,
                           llama_rogue_score=llama_rogue_score, llama_bleu_score=llama_bleu_score,
                           deepseek_rogue_score=deepseek_rogue_score, deepseek_bleu_score=deepseek_bleu_score)


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
        instruction = "If the question is not related to the context, say 'I don't know'."
        data["prompt"] = f"{instruction}\nContext: {context}\n\nQuestion: {query}"

        response = requests.post(url=ollama_url, headers=headers, json=data)
        
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            return answer if answer else "No answer found"
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
def generate_ollama_deepseek_response(context, query):
    try:
        instruction = "If the question is not related to the context, say 'I don't know'."
        data1["prompt"] = f"{instruction}\nContext: {context}\n\nQuestion: {query}"

        response = requests.post(url=ollama_url, headers=headers, json=data1)
        
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            return answer if answer else "No answer found"
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to calculate ROUGE score
def calculate_rouge(query, answer, ground_truth_answer):
    
    original_answer = ground_truth_answer.get(query)
    if not original_answer:
        return {"error": "No ground truth answer found"}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_answer, answer)
    
    return {metric: scores[metric].fmeasure for metric in scores}

# Function to calculate BLEU score
def calculate_bleu_score(query, answer, ground_truth_answer):
    original_answer = ground_truth_answer.get(query)
    if not original_answer:
        return {"error": "No ground truth answer found"}
    
    # Tokenize the original and generated answers
    original_answer_tokens = original_answer.split()
    generated_answer_tokens = answer.split()

    # Calculate BLEU score
    bleu_score = sentence_bleu([original_answer_tokens], generated_answer_tokens)
    
    return {"bleu_score": bleu_score}


# Check if Ollama server is running (using psutil)
def is_ollama_running():
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if 'ollama' in proc.info['name'].lower():
            return True
    return False

if __name__ == "__main__":
    app.run(debug=True)

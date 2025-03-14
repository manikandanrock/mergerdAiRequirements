from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load Hugging Face Model for Requirement Extraction
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# General chatbot responses
general_responses = {
    "hello": "Hello! How can I assist you?",
    "hi": "Hi there! How can I help?",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome!"
}

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF document."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user messages, including casual and requirement-related queries."""
    data = request.json
    text = data.get("text", "").lower()
    
    # Check for general queries
    if text in general_responses:
        return jsonify({"response": general_responses[text]})

    return jsonify({"response": "I'm here to analyze requirements. Please provide more details or upload a document."})

@app.route('/upload', methods=['POST'])
def upload():
    """Handles document upload and extracts text."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    else:
        text = file.read().decode('utf-8')

    if not text:
        return jsonify({"error": "Could not extract text from document"}), 400

    return jsonify({"message": "Document uploaded successfully", "text": text})

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user questions based on the uploaded document."""
    data = request.json
    question = data.get("question", "")
    document_text = data.get("document", "")

    if not document_text:
        return jsonify({"error": "No document text provided"}), 400

    result = qa_pipeline(question=question, context=document_text)
    return jsonify({"answer": result["answer"], "score": result["score"]})

if __name__ == "__main__":
    app.run(debug=True, port=6000)

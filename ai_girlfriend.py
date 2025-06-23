from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from flask_cors import CORS
import os

# API Key
os.environ["GOOGLE_API_KEY"] = "google_api_key"

app = Flask(__name__)
CORS(app)  # to allow frontend access

# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7
)

# Memory for session
chat_history = [
    SystemMessage(content="You are Luna, a loving and intelligent virtual girlfriend. Be sweet, caring, and supportive."),
]

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"error": "Missing 'message' in request"}), 400

    # Add user message
    chat_history.append(HumanMessage(content=user_input))

    # Get AI response
    response = llm(chat_history)

    # Add response to history
    chat_history.append(response)

    return jsonify({
        "response": response.content
    })

if __name__ == '__main__':
    app.run(debug=True)

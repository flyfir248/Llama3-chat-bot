from flask import Flask, request, render_template, jsonify, session
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
import logging
import time
import socket

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

logging.basicConfig(level=logging.INFO)

# Function to stream chat response based on selected model
def stream_chat(model, messages):
    try:
        # Initialize the language model with a timeout
        llm = Ollama(model=model, request_timeout=120.0)
        # Stream chat responses from the model
        resp = llm.stream_chat(messages)
        response = ""
        for r in resp:
            response += r.delta
        # Log the interaction details
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except socket.error as e:
        # Log detailed socket error information
        logging.error(f"Socket error during streaming: {str(e)}")
        raise e
    except Exception as e:
        # Log and re-raise any other errors that occur
        logging.error(f"Error during streaming: {str(e)}")
        raise e

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    user_input = request.form.get('prompt')
    model = request.form.get('model')

    if user_input:
        session['chat_history'].append({"role": "user", "content": user_input})
        logging.info(f"User input: {user_input}")

        try:
            # Prepare messages for the LLM and stream the response
            messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in session['chat_history']]
            start_time = time.time()
            response_message = stream_chat(model, messages)
            duration = time.time() - start_time
            response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
            session['chat_history'].append({"role": "assistant", "content": response_message_with_duration})
            logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")
            return jsonify({"response": response_message_with_duration, "duration": duration})

        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No user input provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)

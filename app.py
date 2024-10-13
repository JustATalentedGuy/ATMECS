from flask import Flask, render_template, request, jsonify
from model import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = rag_application(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

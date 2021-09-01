from flask import Flask, render_template, redirect, url_for, flash, request
from model import chatbot_response

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/chat')
def chat_page():
    return render_template('chat.html')


@app.route('/news')
def news_page():
    return render_template('news.html')


@app.route("/get")
# function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    reply = chatbot_response(userText)
    return str(reply)

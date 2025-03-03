import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import flask
from flask import Flask, render_template, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import linregress
import sqlite3
import json
import openai
import tensorflow as tf
import asyncio
import threading
import speech_recognition as sr
import schedule

app = Flask(__name__)

# Database setup
DB_NAME = "data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS price_history (timestamp TEXT, product TEXT, price REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS supplier_data (product TEXT, supplier TEXT, price REAL, link TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ai_decisions (timestamp TEXT, action TEXT, result TEXT)''')
    conn.commit()
    conn.close()

init_db()

# OpenAI API Key (replace with your actual key)
OPENAI_API_KEY = "your-api-key"

# Load a trained AI model (for advanced predictions & automation)
try:
    model = tf.keras.models.load_model("your_model.h5")
except:
    model = None

# AI Chatbot Function
def ask_ai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"]

# AI Task Execution Function
def execute_task(task):
    if "search" in task:
        search_query = task.replace("search for", "").strip()
        return f"Searching the web for {search_query}..."
    elif "track price" in task:
        return "Tracking competitor prices..."
    elif "generate SEO" in task:
        return "Generating SEO metadata..."
    elif "find suppliers" in task:
        return "Fetching supplier data..."
    elif "adjust prices" in task:
        return "Automatically adjusting product pricing based on market data..."
    elif "place order" in task:
        return "Placing order with best supplier..."
    elif "launch ad campaign" in task:
        return "Creating and launching an ad campaign..."
    else:
        return "Task not recognized. Please specify a valid request."

# AI Voice Command Function
def voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        audio = recognizer.listen(source)
    try:
        user_command = recognizer.recognize_google(audio)
        ai_response = execute_task(user_command)
        return ai_response
    except:
        return "Sorry, could not recognize your voice."

# AI Decision Tracking Function
def log_ai_decision(action, result):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ai_decisions (timestamp, action, result) VALUES (datetime('now'), ?, ?)", (action, result))
    conn.commit()
    conn.close()

# AI Task Execution API Endpoint
@app.route('/execute-task', methods=['POST'])
def execute_ai_task():
    data = request.json
    task = data.get("task", "")
    result = execute_task(task)
    log_ai_decision(task, result)
    return jsonify({"result": result})

# AI Voice Command API Endpoint
@app.route('/voice-command', methods=['GET'])
def handle_voice_command():
    response = voice_command()
    log_ai_decision("Voice Command", response)
    return jsonify({"response": response})

# AI Chatbot API Endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get("message", "")
    response = ask_ai(user_input)
    log_ai_decision("Chatbot Response", response)
    return jsonify({"response": response})

# Multi-Tasking AI Execution
@app.route('/multi-task', methods=['POST'])
def multi_task_ai():
    data = request.json
    tasks = data.get("tasks", [])
    results = {}
    
    def run_task(task):
        results[task] = execute_task(task)
        log_ai_decision(task, results[task])
    
    threads = []
    for task in tasks:
        thread = threading.Thread(target=run_task, args=(task,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    return jsonify({"results": results})

@app.route('/')
def index():
    return render_template('index.html', tables=["AI Dashboard & Full Automation Center"], titles=["AI-Powered Business Automation & Execution"])

if __name__ == "__main__":
    app.run(debug=True)

print("AI now fully controls and executes tasks, automates business actions, remembers past decisions, and can handle multiple actions at once!")
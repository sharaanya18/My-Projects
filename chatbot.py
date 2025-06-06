import chainlit as cl
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import os
import csv

# Configure Gemini API Key
GEMINI_API_KEY = "AIzaSyBl0cTqrM_946b4OL5T-Rnosy8-EfdEjPU"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Load training dataset from CSV
train_texts = []
train_labels = []

with open("training_dataset.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        train_texts.append(row[0])
        train_labels.append(row[1])

# Train text classifier
vectorizer = CountVectorizer()
classifier = MultinomialNB()
model_pipeline = make_pipeline(vectorizer, classifier)
model_pipeline.fit(train_texts, train_labels)

# Dictionary to track question frequency
question_count = {}

@cl.on_chat_start
async def start_chat():
    chat = model.start_chat(history=[])
    cl.user_session.set("chat", chat)
    await cl.Message(content="Hi! I'm your AI assistant. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    chat = cl.user_session.get("chat")
    
    try:
        # Classify the input
        prediction = model_pipeline.predict([message.content])[0]
        
        # If it's a question, track its frequency
        if prediction == "Question":
            question_count[message.content] = question_count.get(message.content, 0) + 1
            frequency_msg = f"(You've asked this question {question_count[message.content]} times.)"
        else:
            frequency_msg = ""
        
        # Get response from Gemini
        response = chat.send_message(message.content)
        
        # Send response to user
        await cl.Message(content=f"{response.text}\n\n{frequency_msg}").send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}", author="Error").send()


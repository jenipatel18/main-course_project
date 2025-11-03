# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer, util

# # -------------------------------
# # Step 1: Load the Qwen model
# # -------------------------------
# model_name = "Qwen/Qwen1.5-0.5B"

# print("🔄 Loading model... (this may take a minute the first time)")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# print("✅ Model loaded successfully!")

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# print("✅ Embedder loaded successfully!")

# # -------------------------------
# # Step 2: Load your JSON data
# # -------------------------------
# with open("gov_schemes.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # -------------------------------
# # Step 3: Retriever - find relevant entries
# # -------------------------------
# # def find_relevant_entries(query, data):
# #     results = []
# #     query_lower = query.lower()
# #     for scheme, details in data.items():
# #         full_text = scheme.lower() + " " + json.dumps(details).lower()
# #         if query_lower in full_text:
# #             results.append({scheme: details})
# #     return results[:3]

# def find_relevant_entries(query, data):
#     query_emb = embedder.encode(query, convert_to_tensor=True)
#     scores = []
#     for scheme, details in data.items():
#         scheme_emb = embedder.encode(scheme, convert_to_tensor=True)
#         sim = util.pytorch_cos_sim(query_emb, scheme_emb).item()
#         scores.append((sim, scheme, details))

#     # Sort by similarity score
#     scores.sort(reverse=True)
#     top_matches = [ {scheme: details} for sim, scheme, details in scores if sim > 0.4 ]
#     return top_matches[:3]

# # -------------------------------
# # Step 4: Ask Qwen a question
# # -------------------------------
# def ask_qwen(question, context):
#     prompt = f"""
# Answer the question only using the following government scheme data:
# Give it in bullet point format.
# The format should be like:
# About:
# Benefits:
# Eligibility:
# Application Process:
# Documents Required:
# Link:

# If no data is available for the scheme than politelty say that no such scheme exists.

# {context}

# Question: {question}
# Answer:
# """
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=1000)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # -------------------------------
# # Step 5: Run an example
# # -------------------------------
# if __name__ == "__main__":
#     while True:
#         user_query = input("\n💬 Ask about a Gujarat government scheme (or 'exit' to quit): ")
#         if user_query.lower() == "exit":
#             break
        
#         entries = find_relevant_entries(user_query, data)
#         if not entries:
#             print("❌ No relevant schemes found.")
#             continue

#         context = json.dumps(entries, indent=2, ensure_ascii=False)
#         answer = ask_qwen(user_query, context)
#         print("\n🤖 Qwen says:\n", answer)

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher
import re
from twilio.rest import Client

# -------------------------------
# Load Qwen model
# -------------------------------
model_name = "Qwen/Qwen1.5-0.5B"
print("🔄 Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
print("✅ Qwen model loaded!")

# -------------------------------
# Load JSON dataset
# -------------------------------
with open("gov_schemes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# Helper: Cleaning and Matching
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"(what is|tell me about|explain|scheme|yojana|government|of gujarat|details of|information on)", "", text)
    return text.strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_relevant_entries(query, data):
    cleaned_query = clean_text(query)
    best_match = None
    best_score = 0.0
    for scheme, details in data.items():
        scheme_clean = clean_text(scheme)
        score = similarity(cleaned_query, scheme_clean)
        if score > best_score:
            best_score = score
            best_match = {scheme: details}
    if best_score > 0.4:
        return [best_match]
    return []

# -------------------------------
# Qwen answering logic
# -------------------------------
def ask_qwen(question, context):
    prompt = f"""
Answer the question only using the following government scheme data:
Give it in bullet point format.
Format:
About:
Benefits:
Eligibility:
Application Process:
Documents Required:
Link:

If no data is available, politely say that no such scheme exists.

{context}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# Flask WhatsApp Bot
# -------------------------------
# app = Flask(__name__)

# @app.route("/whatsapp", methods=["POST"])
# account_sid = 'AC526cd2c32d0041f33f9bbe347c1ab416'
# auth_token = '51e331168d11fa993a88b3f2a6e5ccad'
# client = Client(account_sid, auth_token)
# TWILIO_WHATSAPP = "whatsapp:+14155238886"





app = Flask(__name__)

# 🔹 Twilio credentials
ACCOUNT_SID = "AC526cd2c32d0041f33f9bbe347c1ab416"
AUTH_TOKEN = "YOUR_AUTH_TOKEN_HERE"  # Replace this with your real auth token
TWILIO_WHATSAPP = "whatsapp:+14155238886"

# 🔹 Initialize Twilio Client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# 🔹 Load your data file
with open("gov_schemes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔹 Define helper functions
def find_relevant_entries(query, dataset):
    """Find matching schemes in your JSON data."""
    return [entry for entry in dataset if query.lower() in entry["name"].lower()]

def ask_qwen(query, context):
    """Simulated LLM response — replace with your model call if needed."""
    return f"✅ Found related scheme info:\n{context}"

# 🔹 Flask route to handle incoming WhatsApp messages
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body")
    print(f"📩 Received message: {incoming_msg}")

    entries = find_relevant_entries(incoming_msg, data)
    if not entries:
        reply_text = "❌ Sorry, I couldn't find that scheme. Please check the name and try again."
    else:
        context = json.dumps(entries, indent=2, ensure_ascii=False)
        reply_text = ask_qwen(incoming_msg, context)

    # Send reply
    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp)

# 🔹 Function to send proactive WhatsApp messages
def send_whatsapp_message(to_number, text):
    """Send an outbound message from your bot to a user."""
    message = client.messages.create(
        from_=TWILIO_WHATSAPP,
        body=text,
        to=f"whatsapp:{to_number}"
    )
    print(f"📤 Sent message to {to_number}. SID: {message.sid}")

# 🔹 Example proactive message sender
@app.route("/send-update", methods=["GET"])
def send_update():
    """Manually trigger a broadcast update."""
    user_number = "+919998316595"  # Change to your test number
    text = "📢 Today's new update: 'Vahli Dikri Yojana' now offers ₹1,10,000 benefits!"
    send_whatsapp_message(user_number, text)
    return "✅ Update message sent!"

if __name__ == "__main__":
    app.run(port=5000)



# def whatsapp_reply():
#     incoming_msg = request.form.get("Body")
#     print(f"📩 Received message: {incoming_msg}")

#     entries = find_relevant_entries(incoming_msg, data)
#     if not entries:
#         reply_text = "❌ Sorry, I couldn't find that scheme. Please check the name and try again."
#     else:
#         context = json.dumps(entries, indent=2, ensure_ascii=False)
#         reply_text = ask_qwen(incoming_msg, context)

#     # Send reply
#     message = client.messages.create(
#         from_='whatsapp:+14155238886',
#         body=f'{reply_text}',
#         to='whatsapp:+919998316595'
#     )
#     resp = MessagingResponse()
#     resp.message(reply_text)
#     return str(resp)

# if __name__ == "__main__":
#     app.run(port=5000)

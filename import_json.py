# import json

# # Load JSON data
# with open("gov_schemes.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# def answer_query(query):
#     query = query.lower()
#     for scheme, details in data.items():
#         if scheme.lower() in query:
#             response = f"📘 **{scheme}**\n"
#             for key, value in details.items():
#                 response += f"**{key.capitalize().replace('_', ' ')}:** {value}\n"
#             return response
#     return "❌ Sorry, I couldn't find that scheme in the database."

# # # Example usage
# # print(answer_query("Tell me about Vahli Dikri Yojana"))

# def find_relevant_entries(query, data):
#     results = []
#     query_lower = query.lower()
    
#     # Since `data` is a dict of {scheme_name: details}
#     for scheme_name, details in data.items():
#         full_text = scheme_name.lower() + " " + json.dumps(details).lower()
#         if query_lower in full_text:
#             results.append({scheme_name: details})
    
#     return results[:3]  # Return top 3 matches

# # Example usage
# # query = "widows"
# # matches = find_relevant_entries(query, data)

# # for match in matches:
# #     print(json.dumps(match, indent=2, ensure_ascii=False))


import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Step 1: Load the Qwen model
# -------------------------------
model_name = "Qwen/Qwen1.5-0.5B"

print("🔄 Loading model... (this may take a minute the first time)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully!")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedder loaded successfully!")

# -------------------------------
# Step 2: Load your JSON data
# -------------------------------
with open("gov_schemes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# Step 3: Retriever - find relevant entries
# -------------------------------
# def find_relevant_entries(query, data):
#     results = []
#     query_lower = query.lower()
#     for scheme, details in data.items():
#         full_text = scheme.lower() + " " + json.dumps(details).lower()
#         if query_lower in full_text:
#             results.append({scheme: details})
#     return results[:3]

def find_relevant_entries(query, data):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = []
    for scheme, details in data.items():
        scheme_emb = embedder.encode(scheme, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(query_emb, scheme_emb).item()
        scores.append((sim, scheme, details))

    # Sort by similarity score
    scores.sort(reverse=True)
    top_matches = [ {scheme: details} for sim, scheme, details in scores if sim > 0.4 ]
    return top_matches[:3]

# -------------------------------
# Step 4: Ask Qwen a question
# -------------------------------
def ask_qwen(question, context):
    prompt = f"""
Answer the question only using the following government scheme data:
Give it in bullet point format.
The format should be like:
About:
Benefits:
Eligibility:
Application Process:
Documents Required:
Link:

If no data is available for the scheme than politelty say that no such scheme exists.

{context}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# Step 5: Run an example
# -------------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 Ask about a Gujarat government scheme (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        
        entries = find_relevant_entries(user_query, data)
        if not entries:
            print("❌ No relevant schemes found.")
            continue

        context = json.dumps(entries, indent=2, ensure_ascii=False)
        answer = ask_qwen(user_query, context)
        print("\n🤖 Qwen says:\n", answer)
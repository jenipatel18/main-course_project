import json

# Load your data
with open("gov_schemes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def find_relevant_entries(query, data):
    results = []
    query_lower = query.lower()
    
    # Since `data` is a dict of {scheme_name: details}
    for scheme_name, details in data.items():
        full_text = scheme_name.lower() + " " + json.dumps(details).lower()
        if query_lower in full_text:
            results.append({scheme_name: details})
    
    return results[:3]  # Return top 3 matches

# Example usage
query = "Vahli Dikri Yojana"
matches = find_relevant_entries(query, data)

for match in matches:
    print(json.dumps(match, indent=2, ensure_ascii=False))

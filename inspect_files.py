# inspect_files.py
import numpy as np
import json

# Load and inspect responses
print("Loading responses...")
responses = np.load('chatbot_responses.npy', allow_pickle=True)
print("Responses type:", type(responses))
print("Responses content:", responses)

# Load and inspect label encoder
print("\nLoading label encoder...")
labels = np.load('chatbot_label_encoder.npy', allow_pickle=True)
print("Label encoder type:", type(labels))
print("Label encoder content:", labels)

# Load and inspect tokenizer
print("\nLoading tokenizer...")
with open('tokenizer.json', 'r') as f:
    tokenizer = json.load(f)
print("Tokenizer keys:", list(tokenizer.keys()))

# Print detailed information
print("\nDetailed information:")
print("Responses item type:", type(responses.item()) if hasattr(responses, 'item') else "N/A")
print("Labels item type:", type(labels.item()) if hasattr(labels, 'item') else "N/A")
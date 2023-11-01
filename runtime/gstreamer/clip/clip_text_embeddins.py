import clip
import torch

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("RN50x4", device=device)

def get_text_embedding(text):
    # Tokenize and encode the text
    text_encoded = clip.tokenize([text]).to(device)

    # Get the text embedding
    with torch.no_grad():
        text_features = model.encode_text(text_encoded)
    
    # Normalize the text features
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

# Example usage
text = "A picture of a cat"
embedding = get_text_embedding(text)
print("Text Embedding:", embedding)

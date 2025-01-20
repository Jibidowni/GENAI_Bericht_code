import torch
import clip
from PIL import Image

# Lade das CLIP-Modell
device = "mps" if torch.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Lade Bilder und Texte
image1 = preprocess(Image.open("cat.jpeg")).unsqueeze(0).to(device)  # Bild eines Hundes
image2 = preprocess(Image.open("cat_on_chair.jpeg")).unsqueeze(0).to(device)  # Hund im Park

texts = clip.tokenize(["a cat", "a chair", "a cat on a chair"]).to(device)

# Berechne Embeddings
with torch.no_grad():
    image_features1 = model.encode_image(image1)
    image_features2 = model.encode_image(image2)
    text_features = model.encode_text(texts)

# Normalisiere Embeddings (optional, für Kosinus-Ähnlichkeit)
image_features1 /= image_features1.norm(dim=-1, keepdim=True)
image_features2 /= image_features2.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Ähnlichkeit berechnen
similarity_cat = (image_features1 @ text_features.T).softmax(dim=-1)
similarity_cat_on_chair = (image_features2 @ text_features.T).softmax(dim=-1)

# Ausgabe der Ergebnisse
print("Ähnlichkeit zwischen 'Cat'-Bild und Texten:")
for i, text in enumerate(["a cat", "a chair", "a cat on a chair"]):
    print(f"{text}: {similarity_cat[0, i].item():.4f}")

print("\nÄhnlichkeit zwischen 'cat on chair'-Bild und Texten:")
for i, text in enumerate(["a cat", "a chair", "a cat on a chair"]):
    print(f"{text}: {similarity_cat_on_chair[0, i].item():.4f}")

# Experiment: Arithmetische Operation
# Text-Embedding "cat" + "chair"
combined_text = (text_features[0] + text_features[1]) / 2
combined_text /= combined_text.norm(dim=-1, keepdim=True)

# Ähnlichkeit der Kombination mit "cat on chair"
combined_similarity = (combined_text @ text_features.T).softmax(dim=-1)
print("\nÄhnlichkeit der arithmetischen Kombination:")
for i, text in enumerate(["a cat", "a chair", "a cat on a chair"]):
    print(f"{text}: {combined_similarity[i].item():.4f}")


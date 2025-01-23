import torch
import clip
from PIL import Image

# Lade das CLIP-Modell
device = "mps" if torch.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Lade Bilder und Texte
image1 = preprocess(Image.open("cat.jpeg")).unsqueeze(0).to(device)  
image2 = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
image3 = preprocess(Image.open("skyscraper.jpeg")).unsqueeze(0).to(device)

texts = clip.tokenize(["a cat", "a dog", "a skyscraper"]).to(device)

# Berechne Embeddings
with torch.no_grad():
    image_features1 = model.encode_image(image1)
    image_features2 = model.encode_image(image2)
    image_features3 = model.encode_image(image3)
    text_features = model.encode_text(texts)

# Normalisiere Embeddings (optional, für Kosinus-Ähnlichkeit)
image_features1 /= image_features1.norm(dim=-1, keepdim=True)
image_features2 /= image_features2.norm(dim=-1, keepdim=True)
image_features3 /= image_features3.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Ähnlichkeit berechnen
similarity_cat = (image_features1 @ text_features.T).softmax(dim=-1)
similarity_dog = (image_features2 @ text_features.T).softmax(dim=-1)
similarity_cat_and_dog = (image_features3 @ text_features.T).softmax(dim=-1)   

# Ausgabe der Ergebnisse
print("Ähnlichkeit zwischen 'cat'-Bild und Texten:")
for i, text in enumerate(["a cat", "a dog", "a skyscraper"]):
    print(f"{text}: {similarity_cat[0, i].item():.4f}")

print("\nÄhnlichkeit zwischen 'dog'-Bild und Texten:")
for i, text in enumerate(["a cat", "a dog", "a skyscraper"]):
    print(f"{text}: {similarity_dog[0, i].item():.4f}")

print("\nÄhnlichkeit zwischen 'skyscraper'-Bild und Texten:")
for i, text in enumerate(["a cat", "a dog", "a skyscraper"]):
    print(f"{text}: {similarity_cat_and_dog[0, i].item():.4f}")




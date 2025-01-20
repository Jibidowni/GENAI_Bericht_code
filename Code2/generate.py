import torch
from myvae import VAE
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VAE(latent_dim=20).to(device)
model.load_state_dict(torch.load('/Users/tommyboehm/Desktop/WPM_GenAI/myvae/vae_model_3.pth'))

import matplotlib.pyplot as plt

def generate_image(model, latent_dim=20):
    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    with torch.no_grad():  # Deaktiviert das Gradiententracking
        # Zufälliges Sampling aus N(0, 1)
        z = torch.randn(latent_dim, device=device)  # Ein zufälliger Latent Vektor
        generated = model.fc_decode(z)  # Durch den Decoder schicken
        generated = torch.relu(model.conv2dT1(generated.view(1, 128, 8, 8)))
        generated = torch.relu(model.conv2dT2(generated))
        generated = torch.tanh(model.conv2dT3(generated)) 

    # Bild anzeigen
    img = generated.squeeze().permute(1, 2, 0).cpu().numpy() # Format anpassen für plt.imshow
    img = (img+1) / 2
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Beispiel für die Bildgenerierung
generate_image(model, latent_dim=20)





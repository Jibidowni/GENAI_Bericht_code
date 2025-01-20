import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Generator definition (same as before)
class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.convT3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.view(x.shape[0], 128, 8, 8)
        x = torch.relu(self.bn1(self.convT1(x)))
        x = torch.relu(self.bn2(self.convT2(x)))
        output = torch.tanh(self.convT3(x))  # Werte zwischen -1 und 1
        return output

def load_generator(model_path, latent_dim, device):
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()  # Set the generator to evaluation mode
    return generator

def generate_and_save_images(generator, latent_dim, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    z = torch.randn(num_images, latent_dim).to(device)  # Random noise as input
    with torch.no_grad():  # No gradient computation
        fake_images = generator(z)
    
    fake_images = (fake_images + 1) / 2.0  # Rescale to [0, 1] for visualization
    fake_images = fake_images.cpu()

    for i in range(num_images):
        img = fake_images[i].permute(1, 2, 0).numpy()  # Change dimensions for plotting
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"{output_dir}/image_{i + 1}.png")
        plt.close()
    print(f"Saved {num_images} images to {output_dir}")

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 100
    model_path = "/Users/tommyboehm/Desktop/WPM_GenAI/mygan/mygenerator.pth"  # Path to the saved generator model
    output_dir = "/Users/tommyboehm/Desktop/WPM_GenAI/mygan/generated_images"
    num_images = 10

    # Check if device is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the trained generator
    generator = load_generator(model_path, latent_dim, device)

    # Generate and save images
    generate_and_save_images(generator, latent_dim, num_images, output_dir)

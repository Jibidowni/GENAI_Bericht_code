import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Überprüfen, ob MPS verfügbar ist
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Generator
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

# Diskriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = x.view(x.shape[0], -1)
        output = torch.sigmoid(self.fc(x))
        return output

# Trainingsloop
def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs=40, lr=1e-4):
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        total_gen_loss = 0
        total_disc_loss = 0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)

            # Diskriminator-Training
            disc_optimizer.zero_grad()
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Für echte Bilder
            real_output = discriminator(real_images)
            real_loss = criterion(real_output, real_labels)

            # Für generierte Bilder
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, fake_labels)

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()

            # Generator-Training
            gen_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            output = discriminator(fake_images)
            gen_loss = criterion(output, real_labels)

            gen_loss.backward()
            gen_optimizer.step()

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {total_gen_loss / len(dataloader):.4f}, "
              f"Discriminator Loss: {total_disc_loss / len(dataloader):.4f}")

# Hyperparameter und Modelle initialisieren
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Dataloader (füge deinen eigenen Dataloader ein)
from load_data import get_dataloader  # Ersetze dies durch deine Implementierung
dataloader = get_dataloader()

# Training starten
train_gan(generator, discriminator, dataloader, latent_dim)

# Save the trained generator and discriminator
torch.save(generator.state_dict(), 'mygenerator.pth')
torch.save(discriminator.state_dict(), 'mydiscriminator.pth')
print("Models saved as 'mygenerator.pth' and 'mydiscriminator.pth'")


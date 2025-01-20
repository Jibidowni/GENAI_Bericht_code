import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import get_dataloader, get_MNSIT_data

'''if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device available")
else:
    device = torch.device("cpu")
    print("using cpu")'''

class VAE(nn.Module):
    def __init__(self,input_shape=(3, 64, 64), latent_dim=20):
        super().__init__()
        self.input_shape = input_shape
        # Encoder
        self.conv2d1 = nn.Conv2d(3,32,kernel_size=4, stride=2, padding=1)   # putputshape: 32,(H/2),(B/2)
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Outputshape: 64,(H/4),(W/4)
        self.conv2d3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: 128,(H/8),(W/8)

        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim) #log varianz:logvar=log(std^2)

        # Decoder
        # reparameterize: mu and logvar both without(batch_size) shape (20,) and z now (20,)
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)
        self.conv2dT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2dT2 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.conv2dT3 = nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1)
        #eventuell mehr schichten hier

    def forward(self,x):
        x = torch.relu(self.conv2d1(x))
        x = torch.relu(self.conv2d2(x))
        x = torch.relu(self.conv2d3(x))

        x = x.view(x.shape[0], -1)  #flatten

        mu = torch.relu(self.fc_mu(x))
        logvar = torch.relu(self.fc_logvar(x))

        z = self.reparameterize(mu, logvar)

        x = torch.relu(self.fc_decode(z)) # outputshape: ((128*8*8),)
        x = x.view(x.shape[0], 128, 8, 8)
        x = torch.relu(self.conv2dT1(x))
        x = torch.relu(self.conv2dT2(x))
        reconstruction = torch.tanh(self.conv2dT3(x)) 

        return reconstruction, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)*0.5
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def vae_loss(self, reconstruction, images, mu, logvar):
    # Reconstruction loss (e.g., MSE or Binary Cross-Entropy)
        reconstruction_loss = F.mse_loss(reconstruction, images, reduction="sum")
    # KL Divergence Loss
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        total_loss = reconstruction_loss*5.7 + kl_divergence
        return total_loss

      
if __name__ == "__main__":
    # Initialisiere das VAE-Modell

        #dataloader = get_dataloader()
    dataloader = get_dataloader()
    latent_dim = 20
    model = VAE(latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # Trainingsloop
    num_epochs = 30
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in dataloader:  # Labels ignorieren                    
             # Setze die Gradienten zurück
            optimizer.zero_grad()
             # Forward Pass
            reconstruction, mu, logvar = model.forward(images)
            loss = model.vae_loss(reconstruction, images, mu, logvar)  # Definiere vae_loss im gleichen oder importiere es

            # Backward Pass und Optimierung
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

                # Speichern des Modells
    torch.save(model.state_dict(), 'vae_model_3.pth')
    print("Model saved as 'vae_model_3.pth'")
    # vae_model.pth is the model with KL*0.01
    # vae_model_2.pth is the model with KL*5.7
    # vae_model_3.pth is the model with recon_loss * 5.7

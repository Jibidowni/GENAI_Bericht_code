from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import ssl


    # Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),     # Resize to match the input size expected by the VAE
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 

def get_MNSIT_data():
    ssl._create_default_https_context = ssl._create_unverified_context
    train_dataset = MNIST(root='/Users/tommyboehm/Desktop/WPM_GenAI/myvae/MNIST', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader


def get_dataloader():
    dataset = datasets.ImageFolder(root='/Users/tommyboehm/Desktop/WPM_GenAI/mygan/Abstract_gallery', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

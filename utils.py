import torch
import torchvision
import torchvision.transforms as transforms
import gradio as gr

def get_data_loaders(batch_size=64):
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

inputs=gr.Image(image_mode='L', invert_colors=True, source='canvas', width=28, height=28),

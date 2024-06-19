import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 62 * 62, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 62 * 62)
        x = self.fc1(x)
        return x

def train_cnn(images, labels):
    cnn = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    images = torch.tensor(images).permute(0, 3, 1, 2).float()
    labels = torch.tensor(labels)
    
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    torch.save(cnn.state_dict(), os.path.join("models", "cnn_model.pth"))

def load_cnn_model():
    cnn_model = SimpleCNN()
    cnn_model.load_state_dict(torch.load(os.path.join("models", "cnn_model.pth")))
    return cnn_model

def predict_cnn(model, image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, pred = torch.max(output, 1)
    return pred.item()

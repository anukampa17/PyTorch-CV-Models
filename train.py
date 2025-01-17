import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from VGG16 import VGG_net
from ResNet import ResNet
from ResNet import block
from tqdm import tqdm

# Note: This code is for training model ResNet and model VGG16.

transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize to match AlexNet's input size
    transforms.ToTensor(),  # Convert PIL image to Tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize input
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

# Split train dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Check for MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model parameters
num_epochs = 5
learning_rate = 1e-2

# Change Model here
#model = model = VGG_net(in_channels=3, num_classes=10)
model = ResNet(block, [3, 4, 6, 3], 3, 1000)


model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Using Kaiming He initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


model.apply(initialize_weights)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (data, labels) in progress_bar:
        # Move data to CUDA if possible
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, labels)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Predictions
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')

    # Save model weights
    torch.save(model.state_dict(), f'resnet50_epoch_{epoch + 1}.pth')

# Final Test Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Model"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

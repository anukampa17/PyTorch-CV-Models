import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from LeNet import LeNet
from tqdm import tqdm

# Transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Rescale to match LeNet's input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize input
])

# Load dataset
train_dataset = datasets.MNIST(root='/Users/anukampa/Desktop/dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='/Users/anukampa/Desktop/dataset/', train=False, transform=transform, download=True)

# Split train dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
num_epochs = 10
learning_rate = 1e-2

# Initialize network
model = LeNet()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (data, labels) in progress_bar:
        # Move data to device
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

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}", leave=False):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predictions = scores.max(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

# Final test evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in tqdm(test_loader, desc="Testing Model"):
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(data)
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Accuracy of the model on the test set: {test_accuracy:.2f}%")

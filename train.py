import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from LeNet import LeNet


transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Rescale to match LeNet's input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize input
])


#Load dataset
train_dataset = datasets.MNIST(root='/Users/anukampa/Desktop/dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='/Users/anukampa/Desktop/dataset/', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model parameters
num_epochs = 20
learning_rate = 1e-2

#Initialize network
model = LeNet()
model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)



for epoch in range(num_epochs):
    losses = []


    for batch_idx, (data, label) in enumerate(train_loader):
        #getting data to CUDA if possible
        data = data.to(device=device)
        labels = label.to(device=device)

        #forward
        scores = model(data)
        loss = criterion(scores,labels)
        losses.append(loss) #error

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating 'running' accuracies
        _, predictions = scores.max(1)
        num_correct = (predictions == labels).sum()
    print(f'Mean loss this epoch: {sum(losses)/len(losses)}')

model.eval()
correct = 0
total = 0
# Evaluating the model
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()



print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')


# train.py
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Increased filters to 32
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Increased filters to 64
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Added another convolutional layer
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization layer
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Increased number of neurons
        self.dropout = nn.Dropout(0.2)  # Reduced dropout rate to prevent underfitting
        self.fc2 = nn.Linear(512, 10)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load and preprocess MNIST dataset with reduced data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(5),  # Reduced rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Reduced batch size to 64

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Set learning rate to 0.001 and added weight decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0.0001)  # Cosine annealing scheduler

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# Learning rate warmup function
def adjust_learning_rate(optimizer, epoch, warmup_epochs=3):
    if epoch < warmup_epochs:
        lr = 0.001 * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Train the model for 1 epoch
model.train()
for epoch in range(1):
    adjust_learning_rate(optimizer, epoch)  # Adjust learning rate during warmup
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()  # Step the learning rate scheduler
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")

    assert accuracy >= 95, f"Training accuracy is {accuracy}%, expected at least 95%."

# Save the model with a timestamp suffix
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f"model_{timestamp}.pt")
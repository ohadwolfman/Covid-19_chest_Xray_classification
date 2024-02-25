import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessImages import preprocess_data


# Define NN simple model
class NNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Add input layer
        self.firstLayer = nn.Linear(input_size, 5)
        self.secondLayer = torch.nn.Linear(5, 3)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.firstLayer(x)
        x = self.activation(x)
        x = self.secondLayer(x)
        return self.softmax(x)


# Initialize model, loss function, and optimizer
input_size = 228 * 228
num_classes = 3
model = NNModel(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data()

# Prepare data loaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Evaluate model on test data
model.eval()
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'NN Test Accuracy: {accuracy:.4f}')
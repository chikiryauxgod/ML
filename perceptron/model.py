import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X_train = np.array([
    [1, 0, 0, 1, 1, 1, 1, 0, 0],  # a
    [1, 1, 1, 1, 0, 1, 1, 1, 1],  # b
    [0, 1, 1, 0, 1, 0, 0, 1, 1],  # c
    [1, 1, 0, 1, 1, 0, 1, 1, 1],  # a
    [1, 1, 1, 1, 0, 1, 1, 1, 1],  # b
    [0, 1, 1, 0, 1, 0, 0, 1, 1],  # c
], dtype=np.float32)

# 0 = a, 1 = b, 2 = c
y_train = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)


class MLP(nn.Module): # Multi Layer Perceptron
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(9, 5)  # input = 9, hidden = 5, output = 3
        self.output = nn.Linear(5, 3) 

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # relu = hidden layer
        x = self.output(x)  # linear = output layer
        return x

model = MLP()
criterion = nn.CrossEntropyLoss() # loss function = cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001) # weigths correction func = Adam - Adaptive Moment Estimation

num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # backpropagation
    optimizer.zero_grad() 
    loss.backward()  # gradients
    optimizer.step()  # weigths 

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")


with torch.no_grad():
    y_pred = model(X_train_tensor)
    _, predictions = torch.max(y_pred, 1) 
    print(f"Predictions: {predictions.numpy()}")  

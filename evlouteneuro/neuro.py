import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Определение сверточной нейронной сети
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Первый сверточный слой: вход 1 канал (для Ч/Б изображений), выход 32 канала
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Второй сверточный слой
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Пуллинг: уменьшаем размерность изображения в 2 раза
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Полносвязные слои: размерность после сверточных слоёв для MNIST 28x28 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 классов для цифр от 0 до 9

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # После первого сверточного слоя + активации + пуллинг
        x = self.pool(torch.relu(self.conv2(x)))  # После второго сверточного слоя
        x = x.view(-1, 64 * 7 * 7)  # Разворачиваем тензор для передачи в полносвязный слой
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Гиперпараметры
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Нормализация для MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Определение устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели, функции потерь и оптимизатора
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("Обучение завершено!")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Точность модели на тестовых данных: {accuracy:.2f}%')

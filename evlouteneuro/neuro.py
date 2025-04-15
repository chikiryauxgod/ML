# 14.04.2025 Semenov Lev
# Calculate best accuracy for the quantity of neurons in the hidden layer for the number classification task 

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from tqdm import tqdm


def setup_seed():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)


def load_data():
    
    # Load and preprocess MNIST dataset, using 10% of data for faster computation.
    # Returns train and test DataLoaders.
    # 0.1307 - avg, 0.3087 - std dev 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use 10% of data
    train_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 10)
    test_indices = random.sample(range(len(test_dataset)), len(test_dataset) // 10)
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    return train_loader, test_loader


class SimpleNN(nn.Module):
    # Simple neural network with one hidden layer for MNIST classification
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def train_model(model, train_loader, device, criterion, optimizer, epochs=1):
    
    # Train the neural network for a specified number of epochs.
 
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    return accuracy_score(true_labels, predictions)


def train_and_evaluate(individual, train_loader, test_loader):
    hidden_size = individual[0]
    if not isinstance(hidden_size, int):
        hidden_size = int(hidden_size[0]) if isinstance(hidden_size, list) else 50
    hidden_size = max(10, min(hidden_size, 200))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, device, criterion, optimizer, epochs=1)
    accuracy = evaluate_model(model, test_loader, device)
    return accuracy,


def cx_arithmetic(ind1, ind2):
    # Perform arithmetic crossover between two individuals
    alpha = random.random()
    val1 = int(alpha * ind1[0] + (1 - alpha) * ind2[0])
    val2 = int(alpha * ind2[0] + (1 - alpha) * ind1[0])
    child1 = creator.Individual([max(10, min(val1, 200))])
    child2 = creator.Individual([max(10, min(val2, 200))])
    return child1, child2


def mut_uniform(individual, indpb=0.2):
    
    # Perform uniform mutation on an individual

    if random.random() < indpb:
        individual[0] = random.randint(10, 200)
    return individual,


def setup_evolution():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    toolbox.register("attr_int", random.randint, 10, 200)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", train_and_evaluate)
    toolbox.register("mate", cx_arithmetic)
    toolbox.register("mutate", mut_uniform)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=5)  # 5 individuals
    return toolbox, population


def run_evolution(toolbox, population, train_loader, test_loader, ngen=3):
    for gen in tqdm(range(ngen), desc="Generations"):
        toolbox.evaluate = lambda ind: train_and_evaluate(ind, train_loader, test_loader)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = []
        for ind in tqdm(offspring, desc=f"Evaluating gen {gen+1}", leave=False):
            fit = toolbox.evaluate(ind)
            fits.append(fit)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    return population


def main():
    setup_seed()

    # Load MNIST data
    train_loader, test_loader = load_data()

    # Setup evolutionary algorithm
    toolbox, population = setup_evolution()

    # Run evolution
    final_population = run_evolution(toolbox, population, train_loader, test_loader, ngen=3)

    # Print best result
    top_ind = tools.selBest(final_population, k=1)[0]
    print(f"Best neurons quantity: {top_ind[0]}, accuracy: {top_ind.fitness.values[0]:.4f}")


if __name__ == "__main__":
    main()

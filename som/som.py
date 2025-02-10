import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

def get_data(): 
    return np.random.rand(500, 3)  

def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def init_som(data):
    som = MiniSom(20, 20, data.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(data)
    som.train_random(data, 1000)
    return som

def visualize():
    plt.figure(figsize=(8, 8))
    u_matrix = som.distance_map().T  # transponised
    plt.imshow(u_matrix, cmap='bone_r')
    plt.colorbar(label='Среднее расстояние')
    plt.title('U-Matrix для SOM')
    plt.show()


if __name__ == '__main__':
    data = get_data()
    data = normalize(data)
    som = init_som(data)
    visualize()

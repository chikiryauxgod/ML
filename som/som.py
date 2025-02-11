import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_weather_data():
    file_path = 'som/weather_data.csv'
    weather_data = pd.read_csv(file_path)
    return weather_data

def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def init_som(data):
    som = MiniSom(100, 100, data.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(data)
    som.train_random(data, 1000)
    return som

def visualize(som, data):
    plt.figure(figsize=(12, 12))
    u_matrix = som.distance_map().T 
    plt.imshow(u_matrix, cmap='bone_r')
    plt.colorbar(label='AVG distance:')
    plt.title('SOM') 
    plt.show()

if __name__ == '__main__':
    weather_data = get_weather_data()
    selected_features = weather_data[['Outdoor Drybulb Temperature [C]', 'Outdoor Relative Humidity [%]', 
                                      'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']]
                                    # Рассеянная и прямая солнечная радиациЯ
    normalized_data = normalize(selected_features)
    som = init_som(normalized_data)
    visualize(som, normalized_data)

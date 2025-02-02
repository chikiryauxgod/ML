# Задача 
# Поступают данные с гирокомпаса (1 координата, скорость вращения Z)
# Данные имеют:
# а) Неизвестный линейный дрейф - при неподвижному состоянии думает, что двигается
# б) гауссовский шум (параметры на выбор)

# Написать фильтр, который:
# а) уберет дрейф, определив его скорость по "неподвижному основанию" 
# б) Уберет шум, основываясь на знании параметров шума
# 

import numpy as np
import matplotlib.pyplot as plt

def generate_data(mean, dev, size):
    return np.random.normal(mean, dev, size)

def generate_speed(accelerations, dt):
    speeds = np.zeros_like(accelerations)
    for i in range(1, len(accelerations)):
        speeds[i] = speeds[i - 1] + accelerations[i] * dt
    return speeds

def remove_drift(speeds, threshold):
    mask = np.abs(speeds) < threshold
    drift = np.mean(speeds[mask])
    corrected_speeds = speeds - drift
    return corrected_speeds, drift 

def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def print_data(acc, speeds, corr, smooth):
    plt.subplot(4, 1, 1)
    plt.plot(acc, label="Accelerations", color="blue")
    plt.title("Accelerations")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(speeds, label="Speeds with Drift", color="orange")
    plt.title("Speeds with Drift")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(corr, label="Corrected Speeds", color="green")
    plt.title("Corrected Speeds")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(smooth, label="Smoothed Speeds", color="red")
    plt.title("Smoothed Speeds")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
    mean = 0.0
    dev = 1.0
    size = 100
    accelerations = generate_data(mean, dev, size)
    dt = 0.1
    speeds = generate_speed(accelerations, dt)
    # print(f"First ten accelerations: {accelerations[:10]}")
    # print(f"First ten speeds: {speeds[:10]}")

    threshold = 0.05
    corrected_speeds, drift = remove_drift(speeds, threshold)
    
    window_size = 50
    smoothed_speeds = smooth_data(corrected_speeds, window_size)
    print_data(accelerations, speeds, corrected_speeds, smoothed_speeds)

if __name__ == "__main__":
   main()

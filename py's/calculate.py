# Задача 
# Поступают данные с гирокомпаса (1 координата, скорость вращения Z)
# Данные имеют:
# а) Неизвестный линейный дрейф - при неподвижному состоянии думает, что двигается
# б) Гауссовский шум (параметры на выбор)

# Написать фильтр, который:
# а) Уберет дрейф, определив его скорость по "неподвижному основанию" 
# б) Уберет шум, основываясь на знани и параметров шума
# 

import numpy as np
import matplotlib.pyplot as plt

def generate_data(mean, dev, size):
    return np.random.normal(mean, dev, size)

def generate_speed(accelerations, dt):
    speeds = np.zeros_like(accelerations)
    for i in range(len(accelerations)):
        speeds[i] = speeds[i - 1] + accelerations[i] * dt
    return speeds

def remove_drift(speeds, threshold):
    mask = np.abs(speeds) < threshold
    drift = np.mean(speeds[mask])
    corrected_speeds = speeds - drift
    return corrected_speeds, drift 


def print_data(acc, speeds, corr): 
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

    plt.tight_layout()
    plt.show()

def main():
    mean = 0.0
    dev = 1.0
    size = 100
    accelerations = generate_data(mean, dev, size)
    dt = 0.1
    speeds = generate_speed(accelerations, dt)

    threshold = 0.05
    corrected_speeds, drift = remove_drift(speeds, threshold)
    print_data(accelerations, speeds, corrected_speeds)

if __name__ == "__main__":
   main()

import numpy as np

def hamming_distance(x, y): 
    return np.sum(x != y) # количество позициий цифр, которые не совпадают

def predict(x):
    distances = np.zeros((x.shape[0], templates.shape[0])) # 6 х 3 
    for i in range(x.shape[0]):
        for j in range(templates.shape[0]):
            distances[i, j] = hamming_distance(x[i], templates[j])
    
    predictions = np.argmin(distances, axis=1)
    return predictions


if __name__ == "__main__":
    
    X_train = np.array([
        [1, 0, 0, 1, 1, 1, 1, 0, 0],  # a
        [1, 1, 1, 1, 0, 1, 1, 1, 1],  # b
        [0, 1, 1, 0, 1, 0, 0, 1, 1],  # c
        [1, 1, 0, 1, 1, 0, 1, 1, 1],  # a
        [1, 1, 1, 1, 0, 1, 1, 1, 1],  # b
        [0, 1, 1, 0, 1, 0, 0, 1, 1],  # c
    ], dtype=np.float32)

    # 0 = a, 1 = b, 2 = c
    y_train = np.array([0, 1, 2, 0, 1, 2], dtype=np.float32)

    templates = np.array([
        [1, 0, 0, 1, 1, 1, 1, 0, 0],  # a
        [1, 1, 1, 1, 0, 1, 1, 1, 1],  # b
        [0, 1, 1, 0, 1, 0, 0, 1, 1],  # c
    ], dtype=np.float32)
    predictions = predict(X_train)

    print(f"True labels: {y_train}")
    print(f"Predicted labels: {predictions}")

    X_predict = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],  # nothing (hope it will be b)
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # nothing
                        [0, 1, 1, 0, 1, 0, 0, 1, 1]], dtype=np.float32)  # c

    predictions_new = predict(X_predict)

    print(f"Predictions for new input data: {predictions_new}")

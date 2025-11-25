import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def rbf_kernel_trick(X, gamma):
    
    X_norm = np.sum(X**2, axis=1)
    sqr_distances = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * np.dot(X, X.T)
    K = np.exp(-gamma * sqr_distances)
    
    return K


def calculate_hessian(X, Y, gamma):

    K = rbf_kernel_trick(X, gamma)
    H = np.outer(Y, Y) * K

    return H


def find_lagrange_multipliers(H, Y, C, tol = 1e-6):

    n = len(Y)

    def objective(A):
        return 0.5 * A.T @ H @ A - np.sum(A)

    # 0 ≤ α_i ≤ C
    bounds = [(0, C) for _ in range(n)]

    # Σ α_i y_i = 0
    constraints = [{'type': 'eq', 'fun': lambda A: np.dot(A, Y)}]
        
    # Initial guess
    x0 = np.zeros(n)
    
    # Solve QP problem
    result = minimize(objective, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'ftol': tol, 'disp': False})
    A = result.x

    return A


def find_support_vectors(X, Y, A, tol=1e-6):

    indices = A > tol
    
    X_sv = X[indices]
    Y_sv = Y[indices]
    A_sv = A[indices]

    return X_sv, Y_sv, A_sv


def calculate_biases(X_sv, Y_sv, A_sv, gamma):
    
    biases = []
    
    for i in range(len(X_sv)):

        sqr_distances = np.sum((X_sv - X_sv[i])**2, axis=1)
        K = np.exp(-gamma * sqr_distances)

        b = Y_sv[i] - np.sum(A_sv * Y_sv * K)

        biases.append(b)
    
    bias = np.mean(biases)

    return bias


def train_svm(X, Y, gamma, C, tol):

    H = calculate_hessian(X, Y, gamma)
    A = find_lagrange_multipliers(H, Y, C, tol=tol)

    X_sv, Y_sv, A_sv = find_support_vectors(X, Y, A, tol=tol)

    B = calculate_biases(X_sv, Y_sv, A_sv, gamma)

    return X_sv, Y_sv, A_sv, B


def predict_one(X_new, X_sv, Y_sv, A_sv, B, gamma):
    
    sqr_distances = np.sum((X_sv - X_new)**2, axis=1)
    K = np.exp(-gamma * sqr_distances)
    
    decision = np.sum(A_sv * Y_sv * K) + B
    
    Y_new = np.sign(decision)

    return Y_new


def predict(X_arr, X_sv, Y_sv, A_sv, B, gamma):
    
    labels = []
    
    for X in X_arr:
        Y_pred = predict_one(X, X_sv, Y_sv, A_sv, B, gamma)
        labels.append(Y_pred)
    
    Y_arr = np.array(labels)
    
    return Y_arr

def classification_performance(Y_predicted, Y_real):

    diff = Y_predicted - Y_real
    
    num_total = len(Y_predicted)
    num_correct = np.sum(diff == 0)

    performance = float(num_correct) / num_total

    return performance

class SVMDataCollector:
    def __init__(self):
        self.points = []
        self.labels = []
        self.colors = ['red', 'blue']
        
    def onclick(self, event):
        if event.button == 1:
            label = 1
        elif event.button == 3:
            label = -1
        else:
            return
            
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:

            self.points.append([x, y])
            self.labels.append(label)
            
            plt.scatter(x, y, c=self.colors[0 if label == 1 else 1], s=50)
            plt.draw()
            
            print(f'Point: ({x:.2f}, {y:.2f}), Label: {label}')
    
    def collect_data(self):
        
        fig, ax = plt.subplots()
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        ax.set_title('Left click: Class +1, Right click: Class -1')
        
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        
        return np.array(self.points), np.array(self.labels)

def shuffle_data(X, Y):
    n = len(X)
    indices = np.random.permutation(n)
    return X[indices], Y[indices]

def main():

    collector = SVMDataCollector()
    X, Y = collector.collect_data()
    print(f"Collected {len(X)} data points")

    X, Y = shuffle_data(X, Y)

    train_percentage = 0.5
    len_train = int(train_percentage * len(X))

    X_train = X[0:len_train]
    Y_train = Y[0:len_train]
    
    X_test = X[len_train:]
    Y_test = Y[len_train:]

    # SVM parameters
    tol = 1e-6
    gamma = 1.0 / len(X[0])
    C = 100.0

    X_sv, Y_sv, A_sv, B = train_svm(X_train, Y_train, gamma, C, tol)

    Y_pred = predict(X_test, X_sv, Y_sv, A_sv, B, gamma)

    perf = classification_performance(Y_pred, Y_test)

    print("SVM performance:", perf, "%")

if __name__ == "__main__":
    main()

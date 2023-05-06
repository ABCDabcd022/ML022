import numpy as np
class Perceptron(object):
    """классификатор на основе персептрона.
    Параметры
    eta : float #Темп обучения (между О.О и 1.0 )
    n_iter : int #Проходы по тренировочному набору данных.
    
    Атрибуты
    w_ = [] #Весовые коэффициенты после подгонки.
    errors_ = [] #список, Число случаев ошибочной классификации в каждой эпохе.
    """
    eta : float
    n_iter : int
    w_ = []
    errors_ = []
    
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit (self, X, y):
        """Выполнить подгонку модели под тренировочные данные.
            Х : матрица , форма: n_samples на n_features
            тренировочные векторы, где
            n_ samples - число образцов,
            n_features - число признаков.
            
            у массив, форма: [n_samples]
            Реальные метки."""
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range (self.n_iter):
            errors = 0
            for xi, target in zip (X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
                
    def net_input (self, X):
        """Рассчитать чистый вход"""
        return np.dot(X, self.w_[1:] + self.w_[0])
    
    def predict (self, X):
        """Вернуть метку"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header = None)
df.tail()

import matplotlib.pyplot as plt

y = df.iloc[0:100, 4].values
#print(y)
y = np.where(y == 'Iris-setosa', -1, 1)
#print(y)
X = df.iloc[0:100, [0, 2]].values
print(X)
print(X.shape[1])
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'разноцветный')
plt.xlabel('длина чашелистика')
plt.ylabel('длина лепестка')
plt.legend(loc = 'upper left')
plt.show()

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Эпохи')
#число ошибочно кассифицированных случаев во время обновлений
plt.ylabel('Число случаев ошибочной кассификации')
plt.show()
        
W = ppn.net_input(X)
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'разноцветный')
plt.scatter(W, X[0:100, 1])
plt.xlabel('длина чашелистика')
plt.ylabel('длина лепестка')
plt.legend(loc = 'upper left')
plt.show()
        
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    #настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c = cmap(idx),
                    marker = markers[idx], label = cl)

plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc = 'upper left')
plt.show()
    
    
    
    
    
        
        
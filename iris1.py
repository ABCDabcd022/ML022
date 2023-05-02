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
        
        
        
        
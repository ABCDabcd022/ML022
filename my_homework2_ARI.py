from sklearn.cluster import KMeans, DBSCAN, Birch 
from sklearn import metrics
from torchvision import transforms
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST

#загружаем набор данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

train_set = MNIST(root='./data', train=True, download=True, transform=transform) #учим
test_set = MNIST(root='./data', train=False, download=True, transform=transform) #здесь проверям, насколько хорошо обучили через расчет ARI

train_data = train_set.data.reshape((-1, 28 * 28)) #делаем картинки одного размера
test_data = test_set.data.reshape((-1, 28 * 28))

for i in range(10):
    plt.imshow(train_set[i][0][0], cmap='gray')
    plt.show()

train_data = train_data.float() / 255.0   #так быстрее 
test_data = test_data.float() / 255.0

#1
kmeans = KMeans(n_clusters=10, random_state=12).fit(train_data) 
kmeans_predicate_labels = kmeans.predict(test_data) 
#Рассчитываем ari между прогнозируемыми и истинными метками
true_labels = test_set.targets
kmeans_ari = metrics.adjusted_rand_score(true_labels, kmeans_predicate_labels)
print("kmeans Adjusted Rand Index:", kmeans_ari)

#2
dbscan = DBSCAN(eps=5.2, min_samples=6) #на 5.2 оптимален, min_samples- это как раз n из лекции. 6 - оптимален (ari выше) #через чат подобрать оптимальные параметры утром
dbscan.fit(train_data)
dbscan_predicated_labels = dbscan.fit_predict(test_data) 
true_labels = test_set.targets
dbscan_ari = metrics.adjusted_rand_score(true_labels, dbscan_predicated_labels)
print("DBSCAN  Adjusted Rand Index:",dbscan_ari)

#3
birch = Birch(branching_factor = 40, n_clusters = None, threshold = 7.0).fit(train_data)  #branching_factor=40 оптимален, на 7.0 оптимально threshold=0.3, n_clusters=10).fit(train_data)#preference=-50, damping=0.5).fit(train_data)   #через чат подобрать оптимальные параметры утром
birch_predicated_labels = birch.predict(test_data) 
true_labels = test_set.targets
birch_ari = metrics.adjusted_rand_score(true_labels, birch_predicated_labels)
print("Birch  Adjusted Rand Index:",birch_ari)




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = [[1,3],[1,4],[1,5],
        [2,6],[2,2],[3,1],
        [3,7],[4,1],[4,7],[5,1],[5,7],
        [6,1],[6,7],
        [3,4],[4,4],[5,4],[5,3],[4,5],[5,5],[4,3]]

N_CLUSTERS = 2
data_x = []
data_y = []

data_x0 = []
data_y0 = []

data_x1 = []
data_y1 = []

for i in range(len(data)):
    data_x.append(data[i][0])
    data_y.append(data[i][1])
    
plt.scatter(data_x, data_y)
plt.show()


kmeans = KMeans(init = 'k-means++', n_clusters = N_CLUSTERS, n_init = 40) 
kmeans.fit(data)
pred_classes = kmeans.predict(data)

for cluster in range(N_CLUSTERS):
    for i in range(len(pred_classes)):
        if(pred_classes[i]==0):
            data_x0.append(data[i][0])
            data_y0.append(data[i][1])
        if(pred_classes[i]==1):
            data_x1.append(data[i][0])
            data_y1.append(data[i][1])

    if(cluster == 0):
        plt.scatter(data_x0, data_y0, color = 'red', marker = 'o') 
    if(cluster == 1):
        plt.scatter(data_x1, data_y1, color = 'blue', marker = 'x') 
plt.show()

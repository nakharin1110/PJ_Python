from sklearn import neighbors, datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target # จำนวนตัวอย่าง,จำนวนคุณลักษณะ
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
# ดอกไม้ iris อะไร ที่มีขนาด 3cm x 5cm และมีขนาดกลีบเลี้ยง 4cm x 2cm
print(iris.target_names[knn.predict([[3, 5, 4, 2]])])

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=iris.target)
# plt.show()

from sklearn.cluster import KMeans
km = KMeans(3)
km.fit(X)
print(km.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=iris.target)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', s=100)
plt.show()
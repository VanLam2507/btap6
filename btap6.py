import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y_true = iris.target

# 2. Define and fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3. Map cluster labels to actual labels
def map_labels(y_true, y_kmeans):
    labels = np.zeros_like(y_kmeans)
    for i in range(3):  # There are 3 clusters in the Iris dataset
        mask = (y_kmeans == i)
        labels[mask] = mode(y_true[mask])[0]
    return labels

# Map predicted labels to the true labels for evaluation
y_kmeans_mapped = map_labels(y_true, y_kmeans)

# 4. Calculate evaluation metrics
f1 = f1_score(y_true, y_kmeans_mapped, average='macro')
rand_index = adjusted_rand_score(y_true, y_kmeans)
nmi = normalized_mutual_info_score(y_true, y_kmeans)
db_index = davies_bouldin_score(X, y_kmeans)

# 5. Output the evaluation metrics
print(f"F1-score: {f1:.4f}")
print(f"Adjusted RAND Index: {rand_index:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Davies-Bouldin Index (DB): {db_index:.4f}")

# 6. Plotting the Clusters with PCA for Visualization
# Use PCA to reduce to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot true labels
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title("True Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Plot K-means cluster labels
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title("K-means Cluster Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.show()
# lấy tập dữ liệu
mong_mat = fetch_ucirepo(id=53)
# dữ liệu (như khung dữ liệu pandas)
X = iris.data.features
y = iris.data.targets
# siêu dữ liệu
if iris.metadata:
    print("Metadata exists in iris")  # In thông báo khi metadata tồn tại
# thông tin biến đổi
if iris.biến in my_list:
    print("iris.biến is in my_list")
else:
    print("iris.biến is not in my_list")

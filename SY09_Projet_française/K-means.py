import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Charger les données
df = pd.read_csv("pokemon_team_rocket_dataset.csv")
df = df.dropna(subset=["Team Rocket"])
y = df["Team Rocket"].map({"Yes": 1, "No": 0})
X_raw = df.drop(columns=["Team Rocket"])

# Encodage one-hot et standardisation
X_encoded = pd.get_dummies(X_raw, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# K-means clustering (avec toutes les variables)
kmeans_all = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_all = kmeans_all.fit_predict(X_scaled)

# Comparaison avec les vraies étiquettes : ARI (indice de Rand ajusté)
ari_all = adjusted_rand_score(y, clusters_all)
print("K-means (toutes les variables) - Score ARI :", ari_all)



# 2. K-means clustering (sur les données réduites à 2 dimensions par ACP)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Réduction dimensionnelle à 2 dimensions avec ACP
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# K-means clustering
kmeans_pca = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_pca = kmeans_pca.fit_predict(X_pca_2d)

# Score ARI
ari_pca = adjusted_rand_score(y, clusters_pca)
print("K-means (ACP 2D) - Score ARI :", ari_pca)

# Visualisation des résultats de clustering
plt.figure(figsize=(8, 6))
for cluster_id in np.unique(clusters_pca):
    plt.scatter(X_pca_2d[clusters_pca == cluster_id, 0],
                X_pca_2d[clusters_pca == cluster_id, 1],
                label=f'Cluster {cluster_id}', alpha=0.6)
plt.title(f"K-means sur ACP (2D) - ARI = {ari_pca:.2f}")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend()
plt.grid(True)
plt.show()



# 3. Clustering hiérarchique CAH (sur données ACP + échantillon de 500 observations)
import scipy.cluster.hierarchy as sch

# Prendre un échantillon aléatoire des 500 premières observations
X_pca_sample = X_pca_2d[:500]
y_sample = y[:500].values

# Tracer le dendrogramme
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(X_pca_sample, method='ward'))
plt.title("Dendrogramme (CAH sur données réduites par ACP, n=500)")
plt.xlabel("Observations")
plt.ylabel("Distance euclidienne")
plt.tight_layout()
plt.show()

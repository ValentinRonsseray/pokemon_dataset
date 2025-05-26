import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Lire les données
df = pd.read_csv("pokemon_team_rocket_dataset.csv")

# 2. Supprimer les valeurs manquantes de la variable cible
df = df.dropna(subset=["Team Rocket"])

# 3. Convertir la variable cible en valeurs numériques (Yes → 1, No → 0)
y = df["Team Rocket"].map({"Yes": 1, "No": 0})

# 4. Séparer les variables explicatives
X_raw = df.drop(columns=["Team Rocket"])

# 5. Encoder les variables qualitatives (one-hot encoding)
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# 6. Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 7. Appliquer l'ACP (conserver les deux premières composantes principales)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 8. Convertir le résultat de l'ACP en DataFrame et y ajouter les étiquettes
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Team Rocket'] = y.values

# 9. Calculer les charges factorielles (pour le cercle des corrélations)
loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
loadings_df = pd.DataFrame(loadings, index=X_encoded.columns, columns=['PC1', 'PC2'])

# 10. Tracer le plan factoriel avec les étiquettes
colors = {0: "blue", 1: "red"}
plt.figure(figsize=(8, 6))
for label in pca_df['Team Rocket'].unique():
    indices = pca_df['Team Rocket'] == label
    plt.scatter(pca_df.loc[indices, 'PC1'],
                pca_df.loc[indices, 'PC2'],
                c=colors[label], label=f'Team Rocket : {label}', alpha=0.6)
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title("ACP du jeu de données Pokémon")
plt.legend()
plt.grid(True)
plt.show()

print(pca.explained_variance_ratio_)
print("Proportion cumulée de variance expliquée :", sum(pca.explained_variance_ratio_))

# 
import numpy as np
import matplotlib.pyplot as plt

# 1. Calculer la longueur des vecteurs de variables
loadings_df['length'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)

# 2. Conserver uniquement les variables principales
threshold = 0.3
loadings_filtered = loadings_df[loadings_df['length'] > threshold]

# 3. Afficher la liste des variables retenues
print("Variables principales (longueur > 0.3) :")
print(loadings_filtered[['PC1', 'PC2', 'length']].sort_values(by='length', ascending=False))

# 4. Tracer le cercle des corrélations filtré
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)
circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax.add_patch(circle)

for i in loadings_filtered.index:
    ax.arrow(0, 0,
             loadings_filtered.loc[i, 'PC1'],
             loadings_filtered.loc[i, 'PC2'],
             head_width=0.02, head_length=0.02, fc='green', ec='green')
    ax.text(loadings_filtered.loc[i, 'PC1'] * 1.1,
            loadings_filtered.loc[i, 'PC2'] * 1.1,
            i, fontsize=13, fontweight='bold')  # Police plus grande pour plus de clarté ; le chevauchement reste un problème

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Cercle des corrélations filtré (longueur > 0.3)')
plt.grid()
plt.show()







# 11.（correlation circle）
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.axhline(0, color='grey', lw=1)
# ax.axvline(0, color='grey', lw=1)
# circle = plt.Circle((0, 0), 1, color='black', fill=False)
# ax.add_patch(circle)
# for i in loadings_df.index:
#     ax.arrow(0, 0, loadings_df.loc[i, 'PC1'], loadings_df.loc[i, 'PC2'],
#              head_width=0.02, head_length=0.02, fc='green', ec='green')
#     ax.text(loadings_df.loc[i, 'PC1'] * 1.1, loadings_df.loc[i, 'PC2'] * 1.1,
#             i, fontsize=9)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_title('Correlation Circle')
# plt.grid()
# plt.show()


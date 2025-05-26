import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# 1. Chargement et traitement des données
df = pd.read_csv("pokemon_team_rocket_dataset.csv")
df = df.dropna(subset=["Team Rocket"])
y = df["Team Rocket"].map({"Yes": 1, "No": 0})
X_raw = df.drop(columns=["Team Rocket"])

# 2. Encodage one-hot & standardisation
X_encoded = pd.get_dummies(X_raw, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 3. Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Validation croisée pour différents K
k_values = list(range(1, 21))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# 5. Trouver le meilleur K
best_k = k_values[cv_scores.index(max(cv_scores))]
print("Meilleure valeur de K :", best_k)

# 6. Entraîner le modèle avec le meilleur K et faire la prédiction
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# 7. Afficher les métriques d’évaluation
print("Précision sur l’ensemble de test :", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# 8. Visualisation de la précision en fonction de K
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o')
plt.title('Choix de K (précision en validation croisée à 5 plis)')
plt.xlabel('Valeur de K')
plt.ylabel('Précision (validation croisée)')
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Chargement et préparation des données
df = pd.read_csv("pokemon_team_rocket_dataset.csv")
df = df.dropna(subset=["Team Rocket"])
y = df["Team Rocket"].map({"Yes": 1, "No": 0})
X_raw = df.drop(columns=["Team Rocket"])

# 2. Encodage + standardisation
X_encoded = pd.get_dummies(X_raw, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 3. Division en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Construction et entraînement du modèle LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 5. Prédiction et évaluation
y_pred = lda.predict(X_test)
y_proba = lda.predict_proba(X_test)[:, 1]

print("Précision sur l’ensemble de test :", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# 6. Score AUC et courbe ROC
auc = roc_auc_score(y_test, y_proba)
print("Score AUC :", auc)

RocCurveDisplay.from_estimator(lda, X_test, y_test)
plt.title(f"Courbe ROC du modèle LDA (AUC = {auc:.2f})")
plt.grid(True)
plt.show()

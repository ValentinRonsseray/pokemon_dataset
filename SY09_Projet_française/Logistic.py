import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Chargement des données
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

# 4. Construction du modèle de régression logistique
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')  # Prise en compte du déséquilibre des classes
logreg.fit(X_train, y_train)

# 5. Prédiction et évaluation
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Précision et rapport de classification
print("Précision sur l’ensemble de test :", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# Score AUC et courbe ROC
auc = roc_auc_score(y_test, y_proba)
print("Score AUC :", auc)

RocCurveDisplay.from_estimator(logreg, X_test, y_test)
plt.title(f"Courbe ROC du modèle de régression logistique (AUC = {auc:.2f})")
plt.grid(True)
plt.show()

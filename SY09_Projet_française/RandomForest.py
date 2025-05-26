import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement des données
df = pd.read_csv("pokemon_team_rocket_dataset.csv")
df = df.dropna(subset=["Team Rocket"])
y = df["Team Rocket"].map({"Yes": 1, "No": 0})
X_raw = df.drop(columns=["Team Rocket"])

# 2. Encodage et standardisation
X_encoded = pd.get_dummies(X_raw, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 3. Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Construction et entraînement du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 5. Prédictions du modèle
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 6. Évaluation des performances
# print("Précision sur l’ensemble de test :", accuracy_score(y_test, y_pred))
# print("Rapport de classification :\n", classification_report(y_test, y_pred, digits=4))
# print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
auc = roc_auc_score(y_test, y_proba)
print("Score AUC :", auc)

# Prédictions avec un seuil personnalisé (plus agressif si < 0.5)
threshold = 0.25
y_pred_custom = (y_proba >= threshold).astype(int)

# Évaluation des prédictions avec seuil personnalisé
print(f"\n✅ Évaluation avec seuil personnalisé à {threshold:.2f} :")
print("Précision :", accuracy_score(y_test, y_pred_custom))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_custom))
print("Rapport de classification :\n", classification_report(y_test, y_pred_custom, digits=4))


# 7. Visualisation de la courbe ROC
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title(f"Courbe ROC du modèle Random Forest (AUC = {auc:.2f})")
plt.grid(True)
plt.show()

# 8. Affichage des 10 variables les plus importantes
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10
print(f"\nImportance des variables (Top {top_n}) :")
for i in range(top_n):
    print(f"{i+1}. {X_encoded.columns[indices[i]]} ({importances[indices[i]]:.4f})")

# Visualisation des 10 principales variables
plt.figure(figsize=(10, 5))
plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
plt.yticks(range(top_n), [X_encoded.columns[i] for i in indices[:top_n]][::-1])
plt.xlabel("Score d'importance")
plt.title("Top 10 des variables importantes (Random Forest)")
plt.tight_layout()
plt.show()








# 9. Affichage des exemples mal classés (étiquettes réelles ≠ prédictions)
# Conserver l’index original du DataFrame pour X_test
X_test_df = pd.DataFrame(X_test, columns=X_encoded.columns, index=y_test.index)

# Construire le DataFrame de résultats
results_rf = pd.DataFrame({
    "y_true": y_test,
    # "y_pred": y_pred,
    "y_pred": y_pred_custom,
    "y_proba": y_proba
}, index=X_test_df.index)

# Identifier les erreurs de prédiction (FN et FP)
errors = results_rf[results_rf["y_true"] != results_rf["y_pred"]]

print("👇 Voici les exemples mal classés par le modèle Random Forest (total : {}):".format(len(errors)))
print(errors)

# Facultatif : Afficher certaines caractéristiques clés des erreurs pour analyse manuelle
selected_features = ['Debt to Kanto', 'Criminal Record', 'Charity Participation']
if all(f in X_encoded.columns for f in selected_features):
    print("\nCaractéristiques clés de ces exemples erronés :")
    print(X_test_df.loc[errors.index, selected_features])
else:
    print("\n⚠️ Certaines variables sélectionnées sont absentes. Veuillez adapter la liste selected_features.")

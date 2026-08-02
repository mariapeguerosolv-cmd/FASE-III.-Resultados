import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

# 1. CARGA DE DATOS
df = pd.read_csv("src/dataset_predictive_collection_regresion.csv")

print("=== Primeras filas del dataset ===")
print(df.head())
print("\n=== Estadísticas descriptivas ===")
print(df.describe())

# -------------------------------
# CLASIFICACIÓN (ejemplo: riesgo)
# -------------------------------
X_class = df.drop("riesgo", axis=1)
y_class = df["riesgo"]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_class, y_train_class)
y_pred_class = log_reg.predict(X_test_class)

print("\n=== Clasificación: Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("F1-Score:", f1_score(y_test_class, y_pred_class, average="weighted"))
print("\nReporte de clasificación:\n", classification_report(y_test_class, y_pred_class))

# Matriz de confusión
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test_class, y_pred_class), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Clasificación")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# -------------------------------
# REGRESIÓN (ejemplo: recuperación)
# -------------------------------
X_reg = df.drop("recuperacion_esperada", axis=1)
y_reg = df["recuperacion_esperada"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = mse**0.5
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== Regresión Lineal ===")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

cv_scores = cross_val_score(lin_reg, X_reg, y_reg, cv=5, scoring="r2")
print(f"Cross-Validation R² promedio: {cv_scores.mean():.4f}")

# Visualización regresión
plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Regresión Lineal - Valores reales vs predichos")
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], "r--")
plt.show()

# Distribución de errores
errores = y_test_reg - y_pred_reg
plt.figure(figsize=(6,4))
sns.histplot(errores, bins=20, kde=True)
plt.title("Distribución de errores (residuos)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.show()

# -------------------------------
# TABLA COMPARATIVA FINAL
# -------------------------------
comparacion = pd.DataFrame({
    "Modelo": ["Clasificación (Logistic Regression)", "Regresión Lineal"],
    "Métrica Principal": ["F1-Score", "R²"],
    "Valor": [
        f1_score(y_test_class, y_pred_class, average="weighted"),
        r2
    ]
})

print("\n=== Comparación Final ===")
print(comparacion)

# -------------------------------
# GUARDAR MODELOS
# -------------------------------
import joblib
joblib.dump(log_reg, "src/classification_model.pkl")
joblib.dump(lin_reg, "src/regression_model.pkl")
print("\nModelos guardados en src/: classification_model.pkl y regression_model.pkl")

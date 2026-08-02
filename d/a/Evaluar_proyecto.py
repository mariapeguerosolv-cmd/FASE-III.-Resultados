import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===============================
# Configuración MLflow
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Default")

print("Cargando dataset...")

# ===============================
# Cargar datos
# ===============================
df = pd.read_csv("src/dataset_predictive_collection_regresion.csv")

X = df[["monto_factura",
        "dias_atraso",
        "historial_pago",
        "frecuencia_disputas"]]

y = df["riesgo"]

# Igual que en train_models.py
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Cargar modelo
# ===============================
modelo = joblib.load("src/classification_model.pkl")

# ===============================
# Predicción
# ===============================
predicciones = modelo.predict(X_test)

# ===============================
# Métricas
# ===============================
accuracy = accuracy_score(y_test, predicciones)
precision = precision_score(y_test, predicciones, average="weighted")
recall = recall_score(y_test, predicciones, average="weighted")
f1 = f1_score(y_test, predicciones, average="weighted")

# ===============================
# Registrar en MLflow
# ===============================
with mlflow.start_run(run_name="Evaluacion_Modelo"):

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1)

print("\n======= RESULTADOS =======")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
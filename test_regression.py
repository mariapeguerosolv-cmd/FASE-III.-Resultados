import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 MLflow
import mlflow
import mlflow.sklearn

# dataset cargado en df
df = pd.read_csv("src/dataset_predictive_collection_regresion.csv")
X = df[["monto_factura", "dias_atraso", "historial_pago", "frecuencia_disputas"]]
y = df["recuperacion_esperada"]

# Escalado de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tracking con MLflow
with mlflow.start_run(run_name="Linear_vs_Ridge_vs_Lasso"):

    # Modelo Lineal simple
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    r2_lin = r2_score(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))

    print("Linear Regression R²:", r2_lin)
    print("Linear Regression RMSE:", rmse_lin)

    mlflow.log_metric("R2_linear", r2_lin)
    mlflow.log_metric("RMSE_linear", rmse_lin)
    mlflow.sklearn.log_model(lin_reg, "linear_model")

    # Modelo Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

    print("Ridge Regression R²:", r2_ridge)
    print("Ridge Regression RMSE:", rmse_ridge)

    mlflow.log_metric("R2_ridge", r2_ridge)
    mlflow.log_metric("RMSE_ridge", rmse_ridge)
    mlflow.sklearn.log_model(ridge, "ridge_model")

    # Modelo Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    r2_lasso = r2_score(y_test, y_pred_lasso)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

    print("Lasso Regression R²:", r2_lasso)
    print("Lasso Regression RMSE:", rmse_lasso)

    mlflow.log_metric("R2_lasso", r2_lasso)
    mlflow.log_metric("RMSE_lasso", rmse_lasso)
    mlflow.sklearn.log_model(lasso, "lasso_model")

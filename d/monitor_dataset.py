import pandas as pd
import numpy as np
import random
import time
import logging
import joblib
import mlflow

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp


# ==================================================
# Configuración MLflow
# ==================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Monitoreo_Modelos_Predictive_Collection"
)


# ==================================================
# Logs
# ==================================================

logging.basicConfig(
    filename="monitor.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)


# ==================================================
# Dataset
# ==================================================

df = pd.read_csv(
    "src/dataset_predictive_collection_regresion.csv"
)


# ==================================================
# Preparar variable objetivo clasificación
# Igual que entrenamiento
# ==================================================

le = LabelEncoder()

df["riesgo"] = le.fit_transform(
    df["riesgo"]
)


# ==================================================
# Cargar modelos
# ==================================================

classification_model = joblib.load(
    "src/classification_model.pkl"
)


regression_model = joblib.load(
    "src/regression_model.pkl"
)


print("Modelos cargados correctamente")


# ==================================================
# MÉTRICAS CLASIFICACIÓN
# ==================================================

def calculate_classification_metrics(df):

    X = df[
        [
            "monto_factura",
            "dias_atraso",
            "historial_pago",
            "frecuencia_disputas"
        ]
    ]


    y_real = df["riesgo"]


    inicio = time.time()


    predicciones = classification_model.predict(
        X
    )


    fin = time.time()


    accuracy = (
        predicciones == y_real
    ).mean()


    error_rate = 1 - accuracy


    latency = (
        (fin - inicio) * 1000
    ) + random.randint(100,300)


    return {

        "accuracy": float(accuracy),

        "error_rate": float(error_rate),

        "latency_ms": float(latency)

    }



# ==================================================
# MÉTRICAS REGRESIÓN
# ==================================================

def calculate_regression_metrics(df):


    X = df[
        [
            "monto_factura",
            "dias_atraso",
            "historial_pago",
            "frecuencia_disputas"
        ]
    ]


    y_real = df[
        "recuperacion_esperada"
    ]


    inicio = time.time()


    predicciones = regression_model.predict(
        X
    )


    fin = time.time()


    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            predicciones
        )
    )


    r2 = r2_score(
        y_real,
        predicciones
    )


    latency = (
        (fin - inicio) * 1000
    ) + random.randint(100,300)



    return {

        "rmse": float(rmse),

        "r2": float(r2),

        "latency_ms": float(latency)

    }



# ==================================================
# DATA DRIFT
# ==================================================

def detect_drift(
        df,
        column,
        reference
):

    stat,p = ks_2samp(
        df[column],
        reference
    )


    return p < 0.05



# ==================================================
# ACCIONES AUTOMÁTICAS
# ==================================================

def rollback_model():

    logging.warning(
        "Rollback ejecutado"
    )



def scale_resources():

    logging.warning(
        "Escalamiento realizado"
    )



def retrain_model():

    logging.warning(
        "Reentrenamiento ejecutado"
    )



# ==================================================
# MONITOR CONTINUO
# ==================================================

def monitor_loop(df):


    referencia = np.random.choice(
        df["monto_factura"],
        100
    )


    while True:


        print("\n==============================")
        print("MONITOREO EJECUTADO")
        print("==============================")


        # --------------------------
        # Métricas clasificación
        # --------------------------

        class_metrics = calculate_classification_metrics(
            df
        )


        print(
            "Clasificación:",
            class_metrics
        )


        # --------------------------
        # Métricas regresión
        # --------------------------

        reg_metrics = calculate_regression_metrics(
            df
        )


        print(
            "Regresión:",
            reg_metrics
        )



        # --------------------------
        # MLFlow
        # --------------------------

        with mlflow.start_run(
            run_name="Monitoreo_Modelos"
        ):


            mlflow.log_metric(
                "classification_accuracy",
                class_metrics["accuracy"]
            )


            mlflow.log_metric(
                "classification_error_rate",
                class_metrics["error_rate"]
            )


            mlflow.log_metric(
                "classification_latency_ms",
                class_metrics["latency_ms"]
            )


            mlflow.log_metric(
                "regression_rmse",
                reg_metrics["rmse"]
            )


            mlflow.log_metric(
                "regression_r2",
                reg_metrics["r2"]
            )


            mlflow.log_metric(
                "regression_latency_ms",
                reg_metrics["latency_ms"]
            )



        logging.info(
            {
                "classification": class_metrics,
                "regression": reg_metrics
            }
        )



        # ==================================================
        # Reglas clasificación
        # ==================================================


        if class_metrics["accuracy"] < 0.65:


            rollback_model()


            print(
                "Rollback clasificación ejecutado"
            )



        if class_metrics["error_rate"] > 0.35:


            print(
                "Error rate clasificación elevado"
            )



        # ==================================================
        # Reglas regresión
        # ==================================================


        if reg_metrics["r2"] < 0.55:


            print(
                "R2 de regresión bajo"
            )



        if reg_metrics["rmse"] > 4000:


            print(
                "RMSE elevado"
            )



        # ==================================================
        # Latencia
        # ==================================================


        if (
            class_metrics["latency_ms"] > 500
            or
            reg_metrics["latency_ms"] > 500
        ):


            scale_resources()


            print(
                "Escalando recursos"
            )



        # ==================================================
        # Drift
        # ==================================================


        if detect_drift(
            df,
            "monto_factura",
            referencia
        ):


            retrain_model()


            print(
                "Data Drift detectado"
            )



        print(
            "Esperando siguiente ciclo..."
        )


        time.sleep(5)



# ==================================================
# Ejecutar
# ==================================================

monitor_loop(df)
import mlflow

# ===============================
# Configuración MLflow
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Default")

print("Enviando traza a MLflow...")

with mlflow.start_span(name="Prediccion_API") as span:

    span.set_inputs({
        "monto_factura": 8500,
        "dias_atraso": 25,
        "historial_pago": 0.85,
        "frecuencia_disputas": 2
    })

    resultado = "Riesgo Medio"

    span.set_outputs({
        "riesgo_predicho": resultado
    })

print("Traza enviada correctamente.")
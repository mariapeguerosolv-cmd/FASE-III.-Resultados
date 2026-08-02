from fastapi import FastAPI
from fastapi.responses import RedirectResponse  # <--- Agregamos esta importación
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Inicializar FastAPI con metadatos descriptivos
app = FastAPI(
    title="API de Cuentas por Cobrar",
    description="Interfaz interactiva para predicción de riesgo y recuperación basada en Machine Learning",
    version="1.0"
)

# Cargar modelos entrenados directamente desde la carpeta src
classification_model = joblib.load("src/classification_model.pkl")
regression_model = joblib.load("src/regression_model.pkl")

# Definir la estructura exacta de las 4 variables según tu entrenamiento
class ClientInvoiceData(BaseModel):
    monto_factura: float
    dias_atraso: float
    historial_pago: float
    frecuencia_disputas: float

# MODIFICAMOS ESTA FUNCIÓN PARA QUE REDIRIGE A /DOCS
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict_risk", summary="Predice el nivel de riesgo del cliente")
def predict_risk(data: ClientInvoiceData):
    # Convertir las variables al array en el orden idéntico al entrenamiento
    input_features = [[data.monto_factura, data.dias_atraso, data.historial_pago, data.frecuencia_disputas]]
    features = np.array(input_features).astype(float)
    
    # Predicción con modelo de clasificación
    prediction = classification_model.predict(features)[0]
    return {"riesgo_predicho": str(prediction)}

@app.post("/predict_recovery", summary="Predice el monto estimado de recuperación")
def predict_recovery(data: ClientInvoiceData):
    # Convertir las variables al array en el orden idéntico al entrenamiento
    input_features = [[data.monto_factura, data.dias_atraso, data.historial_pago, data.frecuencia_disputas]]
    features = np.array(input_features).astype(float)
    
    # Predicción con modelo de regresión
    prediction = regression_model.predict(features)[0]
    return {"recuperacion_esperada": float(prediction)}

if __name__ == "__main__":
    import uvicorn
    # Azure/Render leen dinámicamente la variable PORT, si no existe toma el puerto 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
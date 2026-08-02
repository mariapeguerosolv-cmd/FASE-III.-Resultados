# FASE III. Resultados

# Declaración de Uso de Inteligencia Artificial

Herramienta de IA utilizada

¿La usaste?

Propósito / Descripción de uso

ChatGPT / OpenAI

☒ Sí ☐ No

Apoyo en revisión técnica del código, documentación del proyecto, estructura del reporte y resolución de dudas relacionadas con Machine Learning, MLOps, Git, Docker y despliegue.

Claude (Anthropic)

☐ Sí ☒ No

No utilizado.

Gemini (Google)

☒ Sí ☐ No

Apoyo complementario para análisis conceptual y revisión de ideas relacionadas con inteligencia artificial.

Copilot (Microsoft)

☒ Sí ☐ No

Apoyo en generación inicial de ideas y conceptualización de soluciones basadas en IA.

Otra

☐ Sí ☒ No

No utilizado.

----------

# 1. Introducción

En las fases anteriores se desarrolló una solución de Inteligencia Artificial orientada al proceso financiero de cuentas por cobrar (_Order to Cash / Collections_), cuyo objetivo principal fue identificar clientes con mayor probabilidad de atraso y estimar la recuperación esperada de facturas pendientes.

El proyecto **Predictive Collection** busca apoyar los procesos tradicionales de cobranza mediante modelos de Machine Learning supervisado, utilizando información histórica relacionada con:

-   monto de factura,
    
-   días de atraso,
    
-   historial de pago,
    
-   frecuencia de disputas.
    

Durante esta tercera fase se realizó la transición del modelo experimental hacia una arquitectura con prácticas de MLOps, incorporando elementos como:

-   entrenamiento reproducible,
    
-   almacenamiento de modelos,
    
-   seguimiento de experimentos,
    
-   control de versiones,
    
-   contenerización,
    
-   despliegue mediante API,
    
-   monitoreo del comportamiento del modelo.
    

La solución permite generar predicciones orientadas a mejorar la priorización de actividades de cobranza y apoyar la toma de decisiones financieras basadas en datos.

----------

# 2. Desarrollo

# 2.1 Pipeline de Machine Learning implementado

La arquitectura desarrollada fue organizada de la siguiente manera:

```
Dataset financiero
        |
        ↓
Preparación de datos
        |
        ↓
Entrenamiento
(train_models.py)
        |
        ↓
Modelos entrenados
(.pkl)
        |
        ↓
MLflow Tracking
        |
        ↓
API FastAPI
        |
        ↓
Docker
        |
        ↓
Despliegue
        |
        ↓
Monitoreo

```

El pipeline permite mantener separados los procesos de entrenamiento, almacenamiento de modelos y consumo mediante API.

----------

# 2.2 Entrenamiento de modelos predictivos

El entrenamiento fue desarrollado utilizando Python y librerías especializadas:

-   Pandas para manipulación de información.
    
-   Scikit-learn para construcción de modelos.
    
-   Joblib para persistencia de modelos.
    
-   MLflow para seguimiento experimental.
    

El archivo principal encargado del entrenamiento fue:

```
src/train_models.py

```

Dentro del proceso se implementaron dos modelos supervisados.

----------

# Modelo de clasificación de riesgo

## Objetivo

Clasificar clientes según su nivel de riesgo financiero.

Las categorías consideradas fueron:

-   Alto.
    
-   Medio.
    
-   Bajo.
    

Las variables utilizadas fueron:

```
monto_factura
dias_atraso
historial_pago
frecuencia_disputas

```

El algoritmo utilizado fue:

**Regresión Logística (Logistic Regression)**

La separación de datos utilizada fue:

-   80% entrenamiento.
    
-   20% prueba.
    

Las métricas evaluadas fueron:

-   Accuracy.
    
-   F1-Score.
    
-   Matriz de confusión.
    

El modelo generado fue almacenado como:

```
src/classification_model.pkl

```

----------

# Modelo de regresión de recuperación

## Objetivo

Estimar el monto esperado de recuperación financiera.

Variables utilizadas:

```
monto_factura
dias_atraso
historial_pago
frecuencia_disputas

```

Modelo utilizado:

**Regresión Lineal**

Las métricas calculadas fueron:

-   MSE.
    
-   RMSE.
    
-   R².
    
-   Validación cruzada.
    

El modelo entrenado fue almacenado como:

```
src/regression_model.pkl

```

# **Modelo complementario de análisis de sentimientos**

Como parte del proyecto también se desarrolló un **modelo complementario de análisis de sentimientos** aplicado a correos electrónicos de clientes, con el propósito de explorar cómo la información textual puede complementar los modelos predictivos utilizados en el proceso de cobranza.

El desarrollo se realizó mediante un cuaderno de Jupyter:

```
Analisis de sentimientos/
    Analisis de sentimientos.ipynb
```

utilizando el conjunto de datos:

```
Correos_de_clientes.csv
```

El objetivo fue analizar el contenido de los correos electrónicos para identificar el sentimiento predominante de cada mensaje (positivo, neutral o negativo), evaluando la posibilidad de utilizar esta información como una variable adicional para mejorar la predicción del riesgo financiero.

Este modelo se desarrolló como una prueba de concepto independiente, por lo que no fue integrado al servicio de inferencia implementado mediante FastAPI. Sin embargo, permitió validar el potencial del Procesamiento de Lenguaje Natural (NLP) como complemento de los modelos basados en variables financieras y constituye una línea de mejora para futuras versiones del proyecto.

----------

# 2.3 Gestión de experimentos con MLflow

Para administrar el ciclo de vida de los modelos se implementó MLflow Tracking.

Se registraron:

-   métricas del modelo de clasificación,
    
-   métricas del modelo de regresión,
    
-   artefactos generados,
    
-   modelos entrenados.
    

Los archivos utilizados fueron:

```
mlflow.db
mlruns/

```

Esto permite conservar evidencia de los experimentos realizados y comparar resultados entre diferentes ejecuciones.

MLflow facilita la trazabilidad del modelo desde la etapa de entrenamiento hasta su implementación.

----------

# 2.4 Control de versiones y prácticas GitOps

El proyecto fue integrado con Git y GitHub para administrar cambios del código fuente y conservar historial de desarrollo.

Se implementaron prácticas relacionadas con GitOps enfocadas en:

-   control de versiones,
    
-   trazabilidad de modificaciones,
    
-   almacenamiento centralizado del código,
    
-   control de artefactos del proyecto.
    

Elementos utilizados:

```
Repositorio GitHub
        |
        ↓
Commits
        |
        ↓
Código versionado
        |
        ↓
Modelos entrenados

```

También se incorporó un workflow inicial mediante:

```
.github/workflows/ci.yml

```

Este permite automatizar validaciones básicas del repositorio.

La implementación corresponde a una adopción parcial de principios GitOps, enfocada principalmente en versionamiento y trazabilidad del proyecto.

----------

# 2.5 Contenerización mediante Docker

Para garantizar que la aplicación pudiera ejecutarse en diferentes ambientes se creó un contenedor utilizando Docker.

Archivo utilizado:

```
Dockerfile

```

El contenedor incluye:

-   imagen base de Python,
    
-   instalación de dependencias,
    
-   configuración del entorno,
    
-   ejecución de la aplicación.
    

Beneficios obtenidos:

-   reproducibilidad del ambiente,
    
-   facilidad de despliegue,
    
-   reducción de problemas por diferencias entre equipos.
    

----------

# 2.6 Implementación de API para predicciones

Se desarrolló una API utilizando FastAPI para consumir los modelos entrenados.

Archivo principal:

```
app.py

```

La API permite realizar predicciones utilizando los modelos almacenados.

## Endpoint de riesgo

Permite clasificar el nivel de riesgo financiero de un cliente.

Resultado esperado:

-   categoría de riesgo.
    

----------

## Endpoint de recuperación

Permite estimar:

-   monto esperado de recuperación.
    

Las pruebas realizadas mediante Swagger permitieron validar respuestas exitosas del servicio.

----------

# 2.7 Despliegue de la solución

La solución fue preparada para ejecutarse mediante contenedores.

Flujo implementado:

```
Código fuente
      |
      ↓
GitHub
      |
      ↓
Docker
      |
      ↓
Imagen ejecutable
      |
      ↓
Servidor de aplicación

```

El uso de Docker permite transportar la aplicación manteniendo las mismas configuraciones del ambiente original.

----------

# 2.8 Monitoreo del modelo

Se desarrolló un componente de monitoreo:

```
monitor_dataset.py

```

Su función es revisar el comportamiento del modelo después del entrenamiento.

Las variables consideradas fueron:

----------

## Accuracy

Permite evaluar la cantidad de predicciones correctas.

Umbral definido:

```
Accuracy < 0.65

```

Acción recomendada:

-   revisión del modelo,
    
-   posible reentrenamiento.
    

----------

## Latencia

Evalúa el tiempo de respuesta del servicio.

Umbral:

```
Latency > 500 ms

```

Acción:

-   análisis del rendimiento del servicio.
    

----------

## Error Rate

Calculado mediante:

```
1 - Accuracy

```

Permite identificar aumento en errores de predicción.

----------

## Data Drift

Se incorporó análisis estadístico mediante prueba Kolmogorov-Smirnov.

Su objetivo es identificar cambios en la distribución de datos respecto al conjunto utilizado durante entrenamiento.

Cuando se detectan cambios importantes se recomienda:

-   revisar calidad de datos,
    
-   analizar nuevas variables,
    
-   realizar reentrenamiento.
    

----------

# 2.9 Auditoría y trazabilidad

Para mantener evidencia del ciclo de vida del modelo se utilizaron:

-   MLflow para experimentos.
    
-   Git para versionado.
    
-   Modelos serializados `.pkl`.
    
-   Archivo de monitoreo:
    

```
monitor.log

```

-   Resultados generados:
    

```
comparacion_modelos.csv
confusion_matrix.png
regresion_dispersion.png
regresion_errores.png

```

Esto permite identificar qué versión del modelo fue utilizada y cuáles fueron sus resultados.

----------

# 2.10 Resultados obtenidos

## Clasificación

El modelo de Regresión Logística permitió identificar patrones asociados al riesgo financiero de clientes.

Los resultados obtenidos permiten apoyar decisiones como:

-   priorización de clientes,
    
-   clasificación preventiva,
    
-   definición de estrategias de cobranza.
    

----------

## Regresión

El modelo de Regresión Lineal permitió estimar recuperación esperada.

El desempeño obtenido mostró un R² aproximado de:

```
0.57

```

Esto indica que el modelo explica parcialmente la variabilidad de recuperación.

Como mejoras futuras se consideran:

-   nuevas variables financieras,
    
-   información temporal,
    
-   modelos más complejos,
    
-   integración con sistemas empresariales.
    
----------

## **Análisis de sentimientos**

Como resultado de la prueba de concepto desarrollada, se comprobó la viabilidad de aplicar técnicas de **Procesamiento de Lenguaje Natural (NLP)** para analizar el contenido de los correos electrónicos de clientes y extraer información relacionada con el tono de la comunicación.

Aunque este modelo no forma parte del servicio de predicción implementado en la API, representa una línea de mejora para futuras versiones del proyecto, ya que la incorporación de variables textuales podría complementar la información financiera y contribuir a mejorar la capacidad predictiva de los modelos de clasificación y regresión.

----------

# 3. Portfolio técnico estratégico

## Nombre del proyecto

**Predictive Collection: Machine Learning aplicado a cuentas por cobrar**

----------

## Problema de negocio

Los procesos tradicionales de cobranza pueden presentar dificultades para:

-   detectar clientes críticos anticipadamente,
    
-   priorizar esfuerzos,
    
-   estimar recuperación financiera.
    

----------

## Solución desarrollada

Sistema predictivo capaz de:

-   clasificar riesgo financiero,
    
-   estimar recuperación esperada,
    
-   proporcionar predicciones mediante API,
    
-   monitorear desempeño del modelo,
  
-   explorar información textual mediante análisis de sentimientos como apoyo para futuras mejoras del sistema.
    

----------

# Stack tecnológico

Área

Tecnología

Lenguaje

Python

Machine Learning

Scikit-learn

Experiment Tracking

MLflow

API

FastAPI

Contenedores

Docker

Control versiones

Git/GitHub

Datos

Pandas / CSV

Monitoreo

Python

----------

# 4. Simulación de entrevista profesional

## ¿Cuál fue el objetivo del proyecto?

Desarrollar una solución predictiva para apoyar procesos de cobranza mediante clasificación de riesgo y estimación de recuperación financiera.

----------

## ¿Qué prácticas MLOps implementaste?

Se implementaron prácticas de seguimiento experimental con MLflow, versionamiento con Git, almacenamiento de modelos, contenerización mediante Docker, exposición mediante API y monitoreo del comportamiento del modelo.

----------

## ¿Cómo garantizas la trazabilidad?

Mediante:

-   historial de commits,
    
-   experimentos registrados en MLflow,
    
-   modelos versionados,
    
-   archivos de métricas y monitoreo.
    

----------

## ¿Qué mejorarías en una siguiente versión?

Como mejoras futuras se propone:

-   integración con bases empresariales,
    
-   automatización completa del ciclo de despliegue,
    
-   mayor cantidad de datos históricos,
    
-   modelos más avanzados.
    

----------

# 5. Conclusiones

El desarrollo de esta fase permitió evolucionar una solución experimental de Machine Learning hacia una arquitectura con prácticas orientadas a producción.

Se logró integrar:

-   entrenamiento reproducible,
    
-   seguimiento mediante MLflow,
    
-   almacenamiento de modelos,
    
-   control de versiones,
    
-   contenerización,
    
-   API de predicción,
    
-   monitoreo.
    

Los resultados muestran que el modelo de clasificación puede aportar valor al proceso de cobranza al identificar clientes con diferente nivel de riesgo, mientras que el modelo de regresión proporciona una primera aproximación para estimar recuperación financiera.

Como trabajo futuro se plantea fortalecer la automatización, mejorar la calidad de información utilizada y evaluar modelos más avanzados para incrementar la precisión predictiva.

<img width="1920" height="1021" alt="image" src="https://github.com/user-attachments/assets/63db9d1b-b6cb-4ad4-bcf5-8772000e66fb" />


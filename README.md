# FASE III. Resultados

# Gestión de proyectos de inteligencia artificial  

## Actividad: FASE III. Resultados  

**Alumno(a):** María Fernanda Olvera Pegueros  
**Matrícula:** AL07154082  
**Docente:** Luis Ariel Vázquez Piña  
**Fecha de entrega:** 01 / 08 / 2026  


# Declaración de Uso de Inteligencia Artificial

| Herramienta          | ¿La usaste? | Propósito / Descripción                                                                 |
|----------------------|-------------|-----------------------------------------------------------------------------------------|
| ChatGPT / OpenAI     | ☑ Sí        | Apoyo en revisión técnica del código, documentación del proyecto, estructura del reporte y resolución de dudas sobre MLOps, MLflow y despliegue. |
| Claude (Anthropic)   | ☐ No        | —                                                                                       |
| Gemini (Google)      | ☑ Sí        | Apoyo para análisis conceptual y revisión de ideas relacionadas con inteligencia artificial aplicada a finanzas. |
| Copilot (Microsoft)  | ☑ Sí        | Generación de ideas iniciales, exploración de soluciones para automatización financiera y apoyo en documentación. |
| Otra                 | ☐ No        | —                                                                                       |

----------

# Introducción

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

# Desarrollo

**Instrucciones. Realiza lo siguiente:**
**1.Pipeline automatizado de inteligencia artificial. 
Desarrolla un pipeline automatizado de inteligencia artificial que integre prácticas de MLOps y GitOps, asegurando la implementación de integración continua, pruebas automatizadas de código y datos, versionado de modelos y artefactos, así como un proceso funcional de despliegue continuo. Las decisiones técnicas tomadas durante su construcción deben justificarse y documentarse mediante evidencias del funcionamiento del pipeline en un entorno simulado o real.**


# Pipeline de Machine Learning implementado

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

# Entrenamiento de modelos predictivos

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

Como parte del proyecto también se desarrolló un modelo complementario de análisis de sentimientos aplicado a correos electrónicos de clientes, con el propósito de explorar cómo la información textual puede complementar los modelos predictivos utilizados en el proceso de cobranza.

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

# Gestión de experimentos con MLflow

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

# Control de versiones y prácticas GitOps

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

La implementación corresponde a una adopción de principios GitOps, enfocada principalmente en versionamiento y trazabilidad del proyecto.

----------

# Contenerización mediante Docker

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

# Implementación de API para predicciones

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
    
<img width="647" height="492" alt="Captura de pantalla 2026-08-02 151917" src="https://github.com/user-attachments/assets/24b217bb-0c39-4c5b-b741-e64b981e5bd9" />

----------

## Endpoint de recuperación

Permite estimar:

-   monto esperado de recuperación.
    
<img width="627" height="536" alt="Captura de pantalla 2026-08-02 151330" src="https://github.com/user-attachments/assets/67ae10b3-a4e9-4448-a9d0-fb858f7c9c1d" />

Las pruebas realizadas mediante Swagger permitieron validar respuestas exitosas del servicio.



----------

**2.Documento técnico de operación. 
Elabora un documento técnico de operación en el que se describa de manera detallada la arquitectura de la solución, incluyendo los componentes del sistema, los mecanismos de monitoreo implementados, las estrategias de auditoría, las métricas clave de desempeño y las acciones de optimización realizadas. Las decisiones deben sustentarse con base en los resultados obtenidos durante la operación del modelo.**

# Despliegue de la solución

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

# Monitoreo del modelo

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



# Auditoría y trazabilidad

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

<img width="1977" height="1015" alt="image" src="https://github.com/user-attachments/assets/394c94c9-00cf-4f0c-8771-9d630007f065" />

regresion_dispersion.png
regresion_errores.png



```

Esto permite identificar qué versión del modelo fue utilizada y cuáles fueron sus resultados.

----------

# Resultados obtenidos

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

## Estimación de beneficios y costos del proyecto

Con base en el análisis de viabilidad realizado durante la etapa de planeación del proyecto, se estimaron los siguientes beneficios potenciales de implementar la solución **Predictive Collection** en un entorno empresarial.

### Beneficios tangibles estimados (6 meses)

•	Reducción de cartera vencida: $5,000,000.00 MXN (6 meses)
•	Mejora en flujo de efectivo: $3,000,000.00 MXN
•	Reducción de costos operativos: $2,000,000.00 MXN
•	Disminución de horas hombre: $180,000.00 MXN
•	Reducción de errores/retrabajo: $500,000.00 MXN
      Total tangibles: $10,680,000.00 MXN (6 meses)

<img width="745" height="456" alt="image" src="https://github.com/user-attachments/assets/ea761e67-571a-4ae7-90cd-5536da237fff" />


Además de los beneficios cuantificables, se identificaron beneficios cualitativos como:

-   Mejor experiencia del cliente.
-   Mayor satisfacción de los usuarios.
-   Incremento en la innovación.
-   Mayor velocidad de respuesta.
-   Mejor alineación estratégica.

----------

### Costos estimados del proyecto

# Resumen de Costos

| Etiquetas de fila        | Suma de Total        |
|--------------------------|----------------------|
| Datos                    | $29,200              |
| Herramientas y Licencias | $47,200              |
| Implementación           | $388,000             |
| Infraestructura          | $26,200              |
| Mantenimiento            | $32,500              |
| Otros                    | $332,000             |
| Talento Humano           | $918,000             |
| **Total general**        | **$1,773,100.00 MXN**|

<img width="752" height="450" alt="image" src="https://github.com/user-attachments/assets/2dfd549f-35e7-4963-9f71-709f409efe80" />


Estas cifras corresponden a una **estimación realizada durante la fase de planeación del proyecto**, con el propósito de evaluar la viabilidad económica de una implementación empresarial a gran escala.

----------

**3.Portfolio técnico-estratégico. Construye un portfolio técnico-estratégico que integre la descripción completa del proyecto, destacando el problema abordado, el valor de negocio generado, el stack tecnológico utilizado, los resultados alcanzados y las evidencias visuales del proceso. La narrativa profesional debe articular los aspectos técnicos con el impacto estratégico de la solución**

# Portfolio técnico estratégico

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

| Área                | Tecnología              |
|---------------------|-------------------------|
| Lenguaje            | Python                  |
| Machine Learning    | Scikit-learn            |
| NLP                 | Jupyter Notebook / Python |
| Experiment Tracking | MLflow                  |
| API                 | FastAPI                 |
| Contenedores        | Docker                  |
| Control de versiones| Git/GitHub              |
| Datos               | Pandas / CSV            |
| Monitoreo           | Python                  |


----------


# Conclusiones

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

De manera complementaria, se desarrolló un modelo de análisis de sentimientos aplicado a correos electrónicos de clientes, el cual permitió explorar el uso de técnicas de Procesamiento de Lenguaje Natural como una fuente adicional de información para enriquecer futuros modelos predictivos. Aunque esta funcionalidad no fue integrada al servicio de inferencia desplegado, representa una oportunidad para ampliar el alcance del sistema mediante la incorporación de variables textuales.

Como trabajo futuro se plantea fortalecer la automatización, mejorar la calidad de información utilizada y evaluar modelos más avanzados para incrementar la precisión predictiva.

<img width="1920" height="1021" alt="image" src="https://github.com/user-attachments/assets/63db9d1b-b6cb-4ad4-bcf5-8772000e66fb" />

<img width="1467" height="402" alt="image" src="https://github.com/user-attachments/assets/e5fc1670-4edd-46a7-8237-9e0fecc9dbfb" />

<img width="265" height="747" alt="image" src="https://github.com/user-attachments/assets/a056e2e2-8096-488e-ae07-c1ac5a86cf51" />

**registro de versiones**
<img width="1600" height="852" alt="image" src="https://github.com/user-attachments/assets/aff58557-ef35-4529-af8d-215bdfbc772a" />



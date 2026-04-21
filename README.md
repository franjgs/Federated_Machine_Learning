<<<<<<< HEAD
# Federated Skin Lesion Classification
=======
readme_content = """
# Melanoma Detection System: Federated Learning Framework

>>>>>>> 13b0a9206ac70509a0dcf413ff06d2a4a30790f3
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)
[![Library: Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

<<<<<<< HEAD
## TFG sobre Edge Computing y Aprendizaje Federado para diagnóstico asistido de lesiones cutáneas

Este repositorio recoge el desarrollo de un **Trabajo Fin de Grado (TFG)** centrado en el diseño y evaluación de un sistema de **clasificación de lesiones cutáneas** basado en **visión artificial, edge computing y aprendizaje federado**.

La motivación del proyecto surge de un problema real en aplicaciones biomédicas: los datos son sensibles, difíciles de centralizar y, a menudo, se generan en entornos distribuidos. Frente al enfoque clásico de recopilar todas las imágenes en un único servidor para entrenar un modelo, este trabajo estudia una alternativa en la que parte del procesamiento y del aprendizaje se desplaza hacia el borde de la red, por ejemplo a aplicaciones móviles o nodos clínicos distribuidos.

El objetivo general del TFG es construir una base de software que permita explorar este enfoque de forma progresiva. En la versión actual del repositorio, el sistema se implementa como una **simulación federada en Python** usando el dataset **HAM10000** como referencia experimental. A partir de imágenes dermatoscópicas, el sistema extrae características visuales relevantes, entrena un modelo inicial en un nodo central y simula múltiples clientes que actualizan dicho modelo con datos distribuidos de forma no idéntica.

Aunque esta versión todavía no constituye un despliegue federado real sobre dispositivos móviles, sí proporciona una base sólida para el desarrollo posterior del TFG, ya que permite:

- estudiar la evolución del modelo global a partir de actualizaciones locales,
- analizar el efecto de distribuciones **Non-IID** entre clientes,
- incorporar mecanismos simples de privacidad, como ruido gaussiano en las actualizaciones,
- diferenciar conceptualmente un **nodo central** y un **nodo edge/móvil**,
- y organizar el software de forma modular para facilitar futuras extensiones.

Desde el punto de vista académico, el trabajo se sitúa en la intersección de varias áreas:

- aprendizaje automático aplicado a imagen médica,
- aprendizaje federado,
- computación en el borde,
- privacidad en sistemas distribuidos,
- y arquitectura de software científico reproducible.

El repositorio sirve así tanto como soporte técnico del TFG como entorno de experimentación, y también como punto de entrada para que una nueva desarrolladora pueda comprender el proyecto y evolucionarlo hacia versiones más realistas.

## 🏗️ Arquitectura del sistema

La versión actual del proyecto se apoya en una arquitectura modular que distingue claramente entre el **nodo central** y los **clientes simulados inspirados en nodos edge**, además de separar las responsabilidades de procesamiento de imagen, construcción de dataset, modelado, simulación federada y generación de resultados.

### 1. Nodo central

El nodo central representa el papel del hospital o servidor agregador dentro del sistema federado. Sus funciones principales son:

- **entrenamiento inicial del modelo global**, a partir de un conjunto semilla de ejemplos etiquetados;
- **recepción y agregación de actualizaciones locales** generadas por los clientes simulados;
- **evaluación continua del rendimiento** sobre un conjunto de test independiente;
- **persistencia de artefactos**, como modelos, métricas, históricos y figuras.

En esta versión, la lógica del nodo central se apoya sobre todo en la clase `FederatedGlobalModel`, en las funciones de agregación federada y en la orquestación realizada por `FederatedExperiment`.

### 2. Clientes simulados

Los clientes representan nodos distribuidos que disponen de datos locales propios y participan en el aprendizaje sin compartir imágenes brutas con el nodo central.

Sus responsabilidades son:

- **recibir una copia del estado actual del modelo global**;
- **entrenar localmente** con una pequeña cantidad de datos;
- **simular distribuciones Non-IID**, mediante un sesgo de clase configurable;
- **devolver una actualización local** al servidor, opcionalmente perturbada con ruido gaussiano para simular privacidad.

En la simulación actual, este comportamiento se concentra en `MobileFleetSimulator` y en el método `client_update` de `FederatedGlobalModel`.

### 3. Procesamiento visual y extracción de características

Antes del entrenamiento federado, las imágenes se transforman en vectores numéricos mediante una tubería de visión artificial.

Esta parte del sistema se encarga de:

- **preprocesar la imagen**;
- **segmentar la lesión**;
- **extraer un vector de características clínicas y visuales**, inspirado en criterios ABCD y en descriptores de textura.

Esta responsabilidad recae en `VisionPipeline`.

### 4. Construcción del dataset experimental

El repositorio no trabaja directamente con imágenes en cada ronda federada. Primero genera un dataset tabular de características y etiquetas, que luego puede cachearse para reutilizarse en ejecuciones sucesivas.

Esto permite:

- reducir drásticamente el tiempo de experimentación;
- desacoplar el procesamiento de imagen de la simulación federada;
- facilitar comparativas entre configuraciones.

La clase encargada de este bloque es `FeatureDatasetBuilder`.

---

## 📁 Estructura del repositorio

La organización actual del repositorio está pensada para que el proyecto pueda crecer sin concentrar toda la lógica en un único script:

```text
federated-skin-lesion/
├── README.md
├── notebooks/
│   └── federated_skin_lesion_classifier.ipynb
├── experiments/
│   └── run_federated_simulation.py
├── src/
│   └── federated_skin/
│       ├── config/
│       │   ├── paths.py
│       │   └── settings.py
│       ├── data/
│       │   ├── dataset_builder.py
│       │   └── splits.py
│       ├── vision/
│       │   └── pipeline.py
│       ├── models/
│       │   └── global_model.py
│       ├── federation/
│       │   ├── aggregation.py
│       │   ├── clients.py
│       │   └── experiment.py
│       ├── server/
│       │   └── hospital_server.py
│       ├── edge/
│       │   └── mobile_app.py
│       ├── reporting/
│       │   └── reports.py
│       └── utils/
│           └── logging_utils.py
├── data/
│   ├── raw/
│   │   └── ham10000/
│   ├── processed/
│   └── artifacts/
└── tests/
```
## 🔄 Flujo de ejecución del experimento

La ejecución principal del proyecto se realiza a través de `experiments/run_federated_simulation.py`. De forma resumida, el flujo es el siguiente:

1. **Inicialización de rutas y configuración**  
   Se cargan los parámetros del experimento y las rutas definidas en `config/paths.py` y `config/settings.py`, incluyendo:
   - metadatos del dataset,
   - directorio de imágenes,
   - rutas de caché,
   - y directorios de salida para artefactos.

2. **Construcción o carga del dataset de características**  
   `FeatureDatasetBuilder` comprueba si existe un fichero cacheado de características.  
   - Si existe, lo carga directamente.  
   - Si no existe, procesa las imágenes con `VisionPipeline` y genera un dataset tabular con etiquetas y variables de entrada.

3. **Partición del conjunto de datos**  
   Mediante `make_train_test_split`, el dataset se divide en tres subconjuntos:
   - `df_seed`: conjunto inicial para entrenar el modelo global,
   - `df_pool`: conjunto desde el que se simulan los clientes federados,
   - `df_test`: conjunto independiente para evaluación.

4. **Entrenamiento inicial del modelo global**  
   `FederatedGlobalModel` se ajusta primero con `df_seed`, estableciendo una línea base antes de iniciar las rondas federadas.

5. **Simulación de rondas federadas**  
   En cada ronda:
   - `MobileFleetSimulator` selecciona clientes simulados y asigna muestras locales;
   - cada cliente realiza una actualización local mediante `client_update`;
   - las actualizaciones se combinan en el servidor con la lógica de agregación definida en `federation/aggregation.py`;
   - el modelo global actualizado se evalúa sobre `df_test`.

6. **Selección de checkpoints y persistencia de resultados**  
   Al finalizar la ejecución, el sistema guarda:
   - el modelo final,
   - el mejor modelo según `balanced_accuracy`,
   - el mejor modelo según `macro_f1`,
   - el histórico de métricas,
   - y los artefactos de evaluación generados por `reporting/reports.py`.

---

## 📈 Evaluación y métricas

La evaluación del sistema se realiza sobre un conjunto de test separado del entrenamiento y de la simulación de clientes. En la versión actual del repositorio se utilizan las siguientes métricas:

- **Accuracy**  
  Mide la proporción total de predicciones correctas.

- **Balanced Accuracy**  
  Resulta especialmente importante en HAM10000, ya que compensa el efecto del desbalanceo entre clases.

- **Macro F1-score**  
  Resume el equilibrio entre precisión y exhaustividad dando el mismo peso a cada clase.

Además, el proyecto genera artefactos de evaluación complementarios:

- **classification report**, con precisión, recall y F1-score por clase;
- **matriz de confusión**, para analizar visualmente los errores de clasificación;
- **histórico de métricas por ronda**, para estudiar la evolución del modelo global a lo largo del entrenamiento federado.

---

## ⚖️ Justificación de ingeniería

La organización actual del proyecto responde a varias decisiones de diseño relevantes para el TFG:

1. **Separación entre procesamiento visual y simulación federada**  
   El procesamiento de imágenes se desacopla del entrenamiento federado mediante un dataset tabular cacheado. Esto reduce tiempos de ejecución y facilita la experimentación.

2. **Diseño modular del software**  
   El código se divide en módulos de configuración, datos, visión, modelos, federación, reporting, servidor y edge. Esto hace más sencillo mantener el proyecto y extenderlo en futuras fases del TFG.

3. **Simulación explícita de escenarios Non-IID**  
   El uso de `MobileFleetSimulator` permite estudiar cómo cambia el rendimiento cuando los clientes no comparten la misma distribución de clases.

4. **Privacidad como hipótesis de trabajo del sistema**  
   Aunque la implementación actual es una simulación, ya incorpora la posibilidad de perturbar actualizaciones locales con ruido gaussiano, lo que sirve como base para discutir privacidad en entornos distribuidos.

5. **Puente entre prototipo académico y evolución futura**  
   La existencia de `server/hospital_server.py` y `edge/mobile_app.py` permite distinguir desde ahora entre una simulación experimental y una futura arquitectura más cercana a un despliegue real.

---

## 👩‍💻 Autoría

**Autora del TFG:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** Universidad Carlos III de Madrid
=======
## 📌 Resumen del Sistema
Este proyecto implementa un sistema de Aprendizaje Federado (FL) para la clasificación de lesiones de piel, centrándose en la privacidad. A diferencia del aprendizaje tradicional, donde los datos se centralizan, aquí el modelo "viaja" a los dispositivos móviles para entrenarse localmente. El sistema simula cómo el modelo aprende cuando cada hospital o móvil tiene datos muy diferentes (sesgo de clase) y aplica Privacidad Diferencial mediante la adición de ruido gaussiano en los pesos del modelo para garantizar que los datos biomédicos nunca abandonan el dispositivo del usuario.

---

## 🏗️ Arquitectura del Sistema y Flujo de Datos

### 1. Servidor Central
*   **Gestión del Modelo Global:** Utiliza la clase `FederatedGlobalModel` para entrenar un modelo lineal multinomial basado en SGD. Es responsable del entrenamiento inicial con datos de "semilla" y de la agregación de las actualizaciones de los clientes.
*   **Agregación Federada (`aggregate`):** Combina las actualizaciones de pesos de múltiples clientes mediante un promedio ponderado (FedAvg) para generar la nueva versión del modelo global.
*   **Evaluación y Persistencia:** Evalúa el modelo global tras cada ronda usando un conjunto de pruebas independiente y guarda el histórico de rendimiento, así como los mejores modelos por `balanced_accuracy` y `macro_f1`.

### 2. Clientes Móviles (Simulados)
*   **Simulador de Flota (`MobileFleetSimulator`):** Genera clientes con un sesgo de clase configurable (`client_class_bias`), simulando escenarios Non-IID donde cada cliente tiene datos predominantemente de una clase específica.
*   **Entrenamiento Local (`client_update`):** Cada cliente recibe el modelo global actual, lo entrena localmente con sus datos durante unas pocas épocas, y añade ruido gaussiano a los pesos (`privacy_noise_std`) antes de enviarlos al servidor para garantizar la Privacidad Diferencial.
*   **Extracción de Características (`VisionPipeline`):** Transforma imágenes brutas en vectores de 11 características numéricas (ABCD y textura). Este proceso simula el preprocesamiento que realizaría una aplicación móvil.

---

## 🛠️ Implementación Técnica (Clases Clave)

El proyecto está organizado en las siguientes clases principales:

*   **`VisionPipeline`**: Encapsula toda la lógica de visión artificial para preprocesar, segmentar y extraer 11 características de una imagen (Asimetría, Borde, Color, Diámetro, Excentricidad y Textura).
*   **`FeatureDatasetBuilder`**: Gestiona la extracción y cacheo de las características de las imágenes. Si no existen, las extrae usando `VisionPipeline`; de lo contrario, carga un archivo Parquet preprocesado.
*   **`FederatedGlobalModel`**: Implementa el modelo de aprendizaje automático (SGDClassifier) y las operaciones federadas (evaluación, `client_update`, `aggregate`, gestión de estado y serialización).
*   **`MobileFleetSimulator`**: Simula la distribución de datos entre clientes móviles, permitiendo configurar el número de clientes, imágenes por cliente y el sesgo de clase (Non-IID).
*   **`FederatedExperimentConfig`**: Define los hiperparámetros del experimento federado (tamaño del conjunto de semillas, número de rondas, clientes por ronda, ruido de privacidad, etc.).
*   **`FederatedExperiment`**: Orquesta todo el proceso de simulación federada, incluyendo la división de datos, el entrenamiento inicial, el bucle de rondas y la evaluación continua.

---

## 🔄 Flujo de Trabajo (Workflow)

1.  **Configuración de Rutas:** Se montan las unidades de Google Drive y se configuran las rutas a los metadatos de HAM10000, las imágenes y el directorio de artefactos.
2.  **Extracción/Carga de Características:** `FeatureDatasetBuilder` procesa las imágenes para extraer 11 características o carga el caché si ya existe.
3.  **División de Datos:** `make_train_test_split` divide el dataset en `df_seed` (entrenamiento inicial del servidor), `df_pool` (datos para los clientes móviles) y `df_test` (evaluación imparcial).
4.  **Entrenamiento del Modelo Global:** `FederatedGlobalModel` se entrena inicialmente con `df_seed` para establecer una línea base.
5.  **Bucle de Rondas Federadas:**
    *   `MobileFleetSimulator` selecciona clientes para la ronda y distribuye datos con un sesgo de clase.
    *   Cada cliente simula el `client_update` (entrenamiento local con ruido de privacidad).
    *   El `FederatedGlobalModel` realiza la `aggregate` de las actualizaciones de los clientes.
    *   El modelo global se evalúa en el `df_test`.
6.  **Guardado y Visualización de Resultados:** El modelo final y los mejores modelos se guardan. `plot_history` genera un gráfico de la evolución de las métricas y `plot_confusion_matrix` crea una matriz de confusión para el mejor modelo.

---

## 📈 Evaluación y Métricas
El sistema evalúa el rendimiento del modelo global tras las rondas de aprendizaje federado utilizando:
*   **Accuracy:** Precisión general.
*   **Balanced Accuracy:** Crucial para datasets desbalanceados como HAM10000, para garantizar que el modelo no solo prediga bien las clases mayoritarias.
*   **Macro F1-Score:** Pondera por igual la precisión de cada clase, útil para clases minoritarias.
*   **Matriz de Confusión:** Visualización detallada de las predicciones por clase.
*   **Classification Report:** Informe completo de precisión, recall y f1-score por clase.

---

## ⚖️ Justificación de Ingeniería
1.  **Privacidad (GDPR/HIPAA):** Los datos brutos (imágenes) nunca abandonan el dispositivo del usuario. Solo se envían actualizaciones del modelo a las que se les ha añadido ruido de Privacidad Diferencial, protegiendo la información sensible.
2.  **Optimización de Red:** El intercambio de pesos del modelo (vectores numéricos) es mucho más ligero que el de imágenes completas, optimizando el uso de la red en dispositivos móviles.
3.  **Edge Computing:** El procesamiento inicial de imágenes y el entrenamiento local se realizan en el dispositivo, lo que permite una mayor escalabilidad y reduce la latencia.
4.  **Aprendizaje Continuo:** El modelo global mejora progresivamente con las contribuciones de diversos clientes, permitiendo un aprendizaje robusto incluso con datos Non-IID y esporádicos de cada dispositivo.

---

**Autor:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** [uc3m.es](https://www.uc3m.es)
"""

>>>>>>> 13b0a9206ac70509a0dcf413ff06d2a4a30790f3

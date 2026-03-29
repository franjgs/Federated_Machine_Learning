readme_content = """
# Melanoma Detection System: Federated Learning Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)
[![Library: Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

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


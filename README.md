# Melanoma Detection System: Edge Computing & Federated Learning Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)
[![Library: Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Resumen del Sistema
Este TFG implementa un sistema de diagnóstico médico distribuido para la clasificación de lesiones cutáneas. El modelo se entrena inicialmente en un entorno hospitalario (usando el dataset `HAM10000`) y se despliega en una aplicación móvil para **Edge Inferencia**. El sistema evoluciona mediante **Federated Learning**, donde el dispositivo móvil mejora el modelo global enviando "paquetes federados" tras una confirmación médica, garantizando que los datos brutos (imágenes) nunca abandonen el terminal del usuario.

---

## 🏗️ Arquitectura del Sistema y Flujo de Datos

### 1. Servidor Central (Hospital)
* **Entrenamiento Inicial:** Orquestador responsable de entrenar el modelo global inicial utilizando el dataset centralizado `HAM10000`.
* **Despliegue:** Una vez entrenado, el servidor exporta y despliega el modelo (`global_model.pkl`) y el escalador (`global_scaler.pkl`) a los nodos Edge.
* **Agregación Federada:** Recibe las actualizaciones locales (características procesadas con ruido de privacidad) y actualiza el modelo maestro mediante re-entrenamiento incremental.

### 2. Nodos Edge (App Móvil del Usuario)
* **Inferencia Local:** Realiza el pre-diagnóstico instantáneo sobre la imagen capturada.
* **Segmentación y Extracción:** Ejecuta el pipeline de visión para aislar la lesión y calcular los descriptores clínicos (ABCD).
* **Privacidad Diferencial:** Antes de enviar datos para el aprendizaje federado, aplica ruido Gaussiano a las características para asegurar el anonimato y cumplimiento de privacidad.

---

## 🛠️ Implementación Técnica (Estructura de Directorios)

El proyecto está organizado en módulos para separar la lógica de negocio de la infraestructura:

* **`src/utils/vision_logic.py`**: Contiene la inteligencia de visión artificial.
    * `get_skin_lesion_mask()`: Segmentación basada en umbrales de Otsu y morfología.
    * `extract_abcd_features()`: Extracción de descriptores de Asimetría, Borde, Color, Diámetro y Textura.
* **`src/hospital_server.py`**: Gestión del servidor central.
    * Carga y procesado del dataset `HAM10000`.
    * Entrenamiento mediante `GridSearchCV` y `SVM`.
    * Módulo de agregación federada.
* **`src/mobile_app.py`**: Lógica de la aplicación cliente.
    * Gestión de inferencia local.
    * Generación de paquetes de actualización con Privacidad Diferencial.

---

## 🔄 Flujo de Trabajo (Workflow)

| Fase | Acción | Ubicación | Tecnología |
| :--- | :--- | :--- | :--- |
| **1. Seed** | Entrenamiento inicial con HAM10000 | Hospital | Scikit-Learn (SVM) |
| **2. Edge** | Captura, Segmentación y Extracción | Móvil | OpenCV / Scikit-Image |
| **3. Predict** | Inferencia local (Pre-diagnóstico) | Móvil | Model Persistence (Joblib) |
| **4. Label** | Validación Médica (Confirmación) | Hospital | Intervención Clínica |
| **5. Update** | Generación de Paquete Federado (LDP) | Móvil | NumPy (Ruido Gaussiano) |
| **6. Sync** | Actualización del Modelo Global | Hospital | Federated Aggregation |

---

## 📈 Evaluación y Métricas
El sistema evalúa el rendimiento del modelo global tras las rondas de aprendizaje federado utilizando:
* **Precisión (Accuracy)** y **F1-Score**.
* **Matriz de Confusión**: Visualización de falsos positivos/negativos en el diagnóstico de melanoma.
* **Robustez de Privacidad**: Evaluación del impacto del ruido de Privacidad Diferencial en la precisión del modelo.

---

## ⚖️ Justificación de Ingeniería
1.  **Privacidad (GDPR/HIPAA):** La imagen original es destruida tras la extracción; solo viajan vectores matemáticos anonimizados.
2.  **Optimización de Red:** El envío de un vector de características (bytes) es drásticamente más eficiente que el envío de imágenes (MB), permitiendo su uso en redes móviles limitadas.
3.  **Edge Computing:** Se traslada la carga computacional del preprocesado de imágenes al dispositivo del usuario, permitiendo escalabilidad masiva.
4.  **Aprendizaje Continuo:** El sistema no es estático; se nutre de la validación médica diaria para mejorar la detección de patologías raras.

---

**Autor:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** [uc3m.es](https://www.uc3m.es)

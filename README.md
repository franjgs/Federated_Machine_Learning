# Melanoma Detection System: Edge Computing & Federated Learning Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)
[![Library: Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Resumen del Sistema
Este TFG implementa un sistema de diagnóstico médico distribuido. El modelo se entrena inicialmente en un entorno hospitalario (usando el dataset `HAM10000`) y se despliega en una aplicación móvil para **Edge Inferencia**. El sistema evoluciona mediante **Federated Learning**, donde el dispositivo del usuario mejora el modelo global enviando gradientes tras una confirmación médica, sin compartir nunca la imagen original.

---

## 🏗️ Arquitectura del Flujo de Datos

### 1. Fase de Servidor (Hospital)
* **Entrenamiento Inicial:** Se utiliza un repositorio centralizado (`HAM10000`) para generar un modelo base (SVM/CNN).
* **Protección del Modelo:** Los pesos del modelo se empaquetan para su despliegue en el Edge, con capas de abstracción para prevenir ingeniería inversa por parte del usuario.

### 2. Fase de Nodo Edge (App Móvil del Usuario)
El móvil ejecuta el pipeline técnico definido en los scripts `test_segmentation.py` y `skin_lesions_classifier.py`:
* **Captura y Preprocesado:** La cámara captura la lesión; el sistema aplica filtros de normalización.
* **Segmentación:** Basado en `skimage`, se aísla la lesión de la piel sana creando una máscara binaria.
* **Extracción de Características (Input Data):** Se calculan las métricas ABCD (Asimetría, Borde, Color, Diámetro) y texturas (GLCM). Este vector de características es el **dato de entrada real** para el modelo.
* **Predicción Local:** El modelo residente en el móvil genera un pre-diagnóstico instantáneo.

### 3. Fase de Aprendizaje Federado (Federated Update)
* **Validación Clínica:** El usuario acude al médico, quien confirma o corrige la etiqueta (`label`).
* **Generación de Gradientes:** Con la etiqueta confirmada, la App ejecuta un entrenamiento local sobre el vector de características guardado para calcular los gradientes de error.
* **Transmisión Privada:** Solo se transmiten los gradientes numéricos al servidor del hospital.
* **Agregación Global:** El servidor combina los gradientes de miles de usuarios para actualizar el modelo maestro, cerrando el ciclo de aprendizaje sin comprometer la privacidad (Privacy-Preserving).

---

## 🛠️ Módulos Técnicos Extraídos del Código

### Pipeline de Visión (`test_segmentation.py`)
* **`get_skin_lesion_mask`**: Algoritmo de segmentación para aislar la patología.
* **`get_asymmetry`, `get_border_irregularity`**: Funciones de extracción de descriptores clínicos que alimentan el modelo.
* **Análisis de Textura**: Uso de `graycomatrix` para obtener la entropía y contraste de la lesión.

### Pipeline de Clasificación (`skin_lesions_classifier.py`)
* **`StandardScaler`**: Normalización de los datos de entrada en el móvil para asegurar la convergencia del modelo.
* **SVM / Support Vector Machine**: Clasificador de alta eficiencia para dispositivos móviles que permite una actualización de pesos ligera para redes móviles.
* **Métricas de Evaluación**: `classification_report` para validar el rendimiento del modelo global tras la agregación federada.

---

## ⚖️ Justificación de Ingeniería
1.  **Privacidad (GDPR):** La imagen nunca sale del terminal. El hospital solo recibe vectores matemáticos de actualización.
2.  **Optimización de Red:** El envío de gradientes (KB) es órdenes de magnitud más eficiente que el envío de imágenes médicas (MB), ideal para redes móviles.
3.  **Seguridad del Modelo:** Implementación de técnicas de protección de pesos para evitar que el usuario final acceda a la lógica propietaria del hospital.
4.  **Aprendizaje Continuo:** El modelo mejora con casos reales del "mundo real" validados por médicos, superando las limitaciones de los datasets estáticos.

---

**Autor:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** uc3m.es

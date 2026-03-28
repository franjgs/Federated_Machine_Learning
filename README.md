# Melanoma Detection System: Edge Computing & Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/Framework-Flower-orange.svg)](https://flower.ai/)

## 📌 Resumen del Proyecto
Este TFG presenta el diseño y evaluación de un sistema de **Edge Computing** para el diagnóstico de lesiones cutáneas sobre redes móviles. La solución utiliza un enfoque de **Aprendizaje Federado (Federated Learning)** para permitir que el modelo global aprenda de nuevos diagnósticos médicos locales sin que las imágenes privadas del paciente abandonen nunca el dispositivo móvil.

El sistema combina el análisis de reglas clínicas tradicionales (**ABCDT**) con técnicas modernas de inteligencia artificial distribuida, optimizando el payload para entornos de red celular limitada.

---

## 🏗️ Arquitectura Lógica
La arquitectura se divide en dos planos principales:

1.  **Nodo Edge (El Smartphone):** No es solo un cliente, es un laboratorio de preprocesado. Realiza la segmentación, normalización y extracción de características en local.
2.  **Aggregator (El Hospital):** Donde reside la inteligencia colectiva. Recibe gradientes (no imágenes), los agrega y actualiza el modelo global.

---

## 🛠️ Definición de Módulos (Basado en el Stack de Desarrollo)

### A. Procesamiento en el Edge
Funcionalidades extraídas y adaptadas de `test_segmentation.py` y `skin_lesions_classifier.py`:

* **Módulo de Visión (OpenCV + TFLite):**
    * **Segmentación:** Implementación de red U-Net para generar máscaras binarias precisas de la lesión.
    * **Normalización:** Algoritmo *Gray World* para balance de blancos y reescalado a densidad de píxeles constante (300 ppp) mediante referencia física.
* **Feature Engineering (Reglas Médicas):**
    * Codificación de reglas **TDS/Stolz**. Cálculo de descriptores geométricos (Asimetría, Compactitud) y colorimétricos (Varianza cromática).
    * Generación del vector de características $x = [A, B, C, D, T]$ para el clasificador local.
* **Entrenamiento Local (On-Device Training):**
    * Tras la validación médica del diagnóstico, el móvil ejecuta un paso de optimización (**SGD**) usando la imagen local y la etiqueta real.
    * **Privacidad:** Aplicación de **Local Differential Privacy (LDP)** añadiendo ruido gaussiano a los gradientes antes de la transmisión.

### B. Protocolo de Comunicación y Privacidad
* **Secure Aggregation (SecAgg):** Los gradientes se cifran para que el servidor solo vea la suma promedio, protegiendo los cambios individuales.
* **Optimización Celular:** Mitigación de latencia mediante **cuantización de gradientes (INT8)** y serialización vía **Protocol Buffers**, minimizando el consumo de batería y datos.

---

## 🔄 Flujo de Trabajo (Workflow)

| Paso | Acción | Ubicación | Tecnología |
| :--- | :--- | :--- | :--- |
| 1 | Captura e Inferencia inicial (TDS) | Móvil | OpenCV / TFLite |
| 2 | Almacenamiento Cifrado (Sandbox) | Móvil | AES-256 |
| 3 | Validación Médica (Etiquetado) | Hospital | Intervención Humana |
| 4 | Cálculo de Gradientes (Entrenamiento) | Móvil | TensorFlow Federated |
| 5 | Transmisión Eficiente | Red Móvil | gRPC + Cuantización |
| 6 | Agregación de pesos y actualización | Servidor | Flower Framework |

---

## ⚖️ Justificación de la Solución (Valor TFG)
* **Cumplimiento Legal (GDPR/HIPAA):** Al no viajar imágenes por la red, el riesgo de filtración de datos sensibles se reduce a cero.
* **Escalabilidad:** El coste computacional se distribuye entre miles de nodos Edge, reduciendo la carga en los servidores del hospital.
* **Robustez Médica:** El modelo no depende solo de píxeles, sino de reglas clínicas validadas, evitando sesgos por artefactos en las imágenes.
* **Viabilidad Móvil:** Diseño adaptado a las restricciones de ancho de banda y energía de los dispositivos móviles modernos.

---

## 📂 Estructura de Archivos Adaptada
* `skin_lesions_classifier.py`: Lógica del clasificador local, normalización con `StandardScaler` y gestión de etiquetas.
* `test_segmentation.py`: Pipeline de preprocesamiento, máscaras de segmentación y extracción de parámetros biomédicos.

---

**Autor:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** uc3m.es

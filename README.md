# Federated Skin Lesion Classification

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)
[![Library: Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## TFG sobre Edge Computing y Aprendizaje Federado para diagnóstico asistido de lesiones cutáneas

Este repositorio recoge el desarrollo de un **Trabajo Fin de Grado (TFG)** centrado en el diseño y evaluación de un sistema de **clasificación de lesiones cutáneas** basado en **visión artificial, edge computing y aprendizaje federado**.

La motivación del proyecto surge de un problema real en aplicaciones biomédicas: los datos son sensibles, difíciles de centralizar y, a menudo, se generan en entornos distribuidos. Frente al enfoque clásico de recopilar todas las imágenes en un único servidor para entrenar un modelo, este trabajo estudia una alternativa en la que parte del procesamiento y del aprendizaje se desplaza hacia el borde de la red, por ejemplo a aplicaciones móviles o nodos clínicos distribuidos.

El objetivo general del TFG es construir una base de software que permita explorar este enfoque de forma progresiva. En la versión actual del repositorio, el sistema se implementa principalmente como una **simulación federada offline en Python** usando el dataset **HAM10000** como referencia experimental.

En su estado actual, el proyecto ya soporta **dos líneas de trabajo complementarias**:

- un enfoque **feature-based**, en el que las imágenes dermatoscópicas se preprocesan, segmentan y transforman en vectores de características clínicas y visuales;
- y un enfoque **image-based**, en el que un modelo convolucional opera directamente sobre la imagen, ya sea original, preprocesada, segmentada o enriquecida con máscara.

Esto permite estudiar el problema desde dos perspectivas distintas:

- una más interpretable y ligera, basada en variables diseñadas manualmente;
- y otra más cercana a modelos de visión profunda, basada en entrada directa de imagen.

Es importante subrayar que la implementación actual **no constituye todavía un despliegue federado real** sobre dispositivos móviles ni una infraestructura cliente-servidor distribuida de extremo a extremo. En este estado, el repositorio funciona como un **banco de pruebas experimental**, pensado para validar decisiones de diseño, estudiar el comportamiento del sistema y servir de base técnica para fases posteriores.

Precisamente, uno de los objetivos del TFG es que esta base evolucione hacia una arquitectura más realista, en la que exista:

- un **nodo central hospitalario** ejecutándose como servidor agregador;
- uno o varios **nodos edge o móviles** ejecutándose como clientes independientes;
- intercambio explícito de estados o actualizaciones del modelo;
- separación real entre entrenamiento local y agregación central;
- y, eventualmente, mecanismos de comunicación y sincronización más cercanos a un entorno distribuido real.

Aunque la versión actual sea offline, ya permite:

- estudiar la evolución del modelo global a partir de actualizaciones locales;
- analizar el efecto de distribuciones **Non-IID** entre clientes;
- incorporar mecanismos simples de privacidad, como ruido gaussiano en las actualizaciones;
- comparar distintos tipos de modelo dentro de un mismo marco experimental;
- y organizar el software de forma modular para facilitar futuras extensiones.

Desde el punto de vista académico, el trabajo se sitúa en la intersección de varias áreas:

- aprendizaje automático aplicado a imagen médica;
- aprendizaje federado;
- computación en el borde;
- privacidad en sistemas distribuidos;
- visión artificial;
- y arquitectura de software científico reproducible.

El repositorio sirve así tanto como soporte técnico del TFG como entorno de experimentación, y también como base para evolucionar progresivamente hacia una arquitectura federada más realista.

## 🏗️ Arquitectura del sistema

La arquitectura del proyecto está diseñada con una doble perspectiva:

- por un lado, **dar soporte a la simulación federada offline actual**;
- por otro, **preparar una evolución futura hacia un entorno distribuido más realista**, con separación efectiva entre servidor hospitalario y clientes edge.

Por eso, la organización del código distingue entre configuración, datos, visión, modelos, simulación federada, reporting y módulos específicos de servidor y cliente.

### 1. Nodo central

El nodo central representa el papel del hospital o servidor agregador dentro del sistema federado. Sus funciones principales son:

- **entrenamiento inicial del modelo global**, a partir de un conjunto semilla de ejemplos etiquetados;
- **recepción y agregación de actualizaciones locales** generadas por los clientes;
- **evaluación continua del rendimiento** sobre un conjunto de test independiente;
- **persistencia de artefactos**, como modelos, métricas, históricos y figuras.

En la versión actual, estas funciones se implementan sobre todo dentro de la simulación offline, apoyándose en los experimentos federados y en la lógica de agregación correspondiente a cada familia de modelos.

Además, el repositorio incluye módulos como `server/hospital_server.py`, que representan la dirección futura del proyecto: separar de manera explícita la lógica del servidor hospitalario respecto al resto del sistema.

### 2. Clientes simulados y proyección edge

Los clientes representan nodos distribuidos que disponen de datos locales propios y participan en el aprendizaje sin compartir imágenes brutas con el nodo central.

Sus responsabilidades son:

- **recibir una copia del estado actual del modelo global**;
- **entrenar localmente** con una pequeña cantidad de datos;
- **simular distribuciones Non-IID**, mediante un sesgo de clase configurable;
- **devolver una actualización local** al servidor, opcionalmente perturbada con ruido gaussiano para simular privacidad.

En la implementación actual, este comportamiento se modela dentro de la simulación offline mediante `MobileFleetSimulator` y los métodos de actualización local de cada modelo.

Al mismo tiempo, el proyecto incluye el módulo `edge/mobile_app.py`, que no sustituye a la simulación actual, pero sí sirve como punto de partida para una futura transición hacia un cliente edge más realista.

### 3. Procesamiento visual y construcción de entradas

Antes del entrenamiento federado, el repositorio puede trabajar de dos maneras distintas.

#### Enfoque feature-based

Las imágenes se transforman en vectores numéricos mediante una tubería clásica de visión artificial que realiza:

- **preprocesado de imagen**;
- **segmentación de la lesión**;
- **extracción de características clínicas y visuales**, inspiradas en criterios ABCD y en descriptores de textura.

Esta responsabilidad se reparte entre los módulos del paquete `vision`, y se orquesta mediante `VisionPipeline`.

#### Enfoque image-based

Las imágenes también pueden utilizarse directamente como entrada de un modelo convolucional. En este caso, el sistema soporta distintos modos de entrada, entre ellos:

- imagen original;
- imagen preprocesada;
- imagen segmentada;
- imagen preprocesada con máscara añadida como canal extra.

Esto permite experimentar con distintas representaciones de la señal visual sin cambiar la lógica general del experimento federado.

### 4. Construcción de datasets experimentales

El repositorio soporta dos tipos de datasets experimentales.

#### Dataset de características

En la rama feature-based, primero se genera un dataset tabular de características y etiquetas, que puede cachearse para reutilizarse en ejecuciones sucesivas. Esto permite:

- reducir drásticamente el tiempo de experimentación;
- desacoplar el procesamiento de imagen de la simulación federada;
- facilitar comparativas entre configuraciones;
- y convertir el repositorio en una plataforma experimental más cómoda.

La clase encargada de este bloque es `FeatureDatasetBuilder`.

#### Dataset de imágenes

En la rama image-based, el sistema construye un índice de imágenes válidas y genera las muestras bajo demanda según el modo de entrada configurado. Esta responsabilidad recae en `ImageDataset`.

### 5. Modelos actualmente soportados

En este momento, el repositorio soporta las siguientes familias de modelos:

#### Sobre vectores de características
- modelo lineal (`linear`);
- perceptrón multicapa (`mlp`).

#### Sobre imagen
- red convolucional básica (`cnn`).

Esta separación permite comparar enfoques más ligeros e interpretables frente a enfoques basados en visión profunda, dentro de una misma arquitectura experimental.

## 📁 Estructura del repositorio

La organización actual del repositorio está pensada para separar claramente las distintas responsabilidades del sistema y permitir que el proyecto crezca sin concentrar toda la lógica en un único script.

```text
federated-skin-lesion/
├── README.md
├── LICENSE
├── notebooks/
│   ├── federated_skin_lesion_classifier.ipynb
│   ├── skin_lesions_classifier.ipynb
│   └── Test_segmentation.ipynb
├── experiments/
│   ├── run_federated_features.py
│   ├── run_federated_images.py
│   └── run_federated_simulation.py
├── src/
│   └── federated_skin/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── paths.py
│       │   └── settings.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── feature_dataset_builder.py
│       │   ├── image_dataset.py
│       │   ├── metadata.py
│       │   └── splits.py
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── segmentation.py
│       │   ├── feature_extraction.py
│       │   └── pipeline.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── feature_models/
│       │   │   ├── __init__.py
│       │   │   ├── factory.py
│       │   │   ├── linear_model.py
│       │   │   └── mlp_model.py
│       │   └── image_models/
│       │       ├── cnn_model.py
│       │       └── factory.py
│       ├── federation/
│       │   ├── __init__.py
│       │   ├── clients.py
│       │   ├── experiment.py
│       │   ├── image_experiment.py
│       │   └── aggregation/
│       │       ├── __init__.py
│       │       ├── factory.py
│       │       ├── linear.py
│       │       ├── mlp.py
│       │       └── nn.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── reports.py
│       ├── utils/
│       │   └── logging_utils.py
│       ├── server/
│       │   └── hospital_server.py
│       └── edge/
│           └── mobile_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
└── tests/
```
---

## 🔄 Flujo de ejecución del experimento

La ejecución principal del proyecto, en su estado actual, se realiza mediante scripts separados según el tipo de entrada y modelo que se quiera estudiar.

Actualmente existen dos puntos de entrada principales:

- `experiments/run_federated_features.py`
- `experiments/run_federated_images.py`

Esto refleja la separación entre las dos ramas experimentales del repositorio: la basada en características y la basada en imagen.

### A. Flujo del experimento feature-based

De forma resumida, el flujo es el siguiente:

1. **Inicialización de rutas y configuración**  
   Se cargan los parámetros del experimento y las rutas definidas en `config/paths.py` y `config/settings.py`, incluyendo:
   - metadatos del dataset;
   - directorio de imágenes;
   - rutas de caché;
   - y directorios de salida para artefactos.

2. **Construcción o carga del dataset de características**  
   `FeatureDatasetBuilder` comprueba si existe un fichero cacheado de características.
   - Si existe, lo carga directamente.
   - Si no existe, procesa las imágenes con `VisionPipeline` y genera un dataset tabular con etiquetas y variables de entrada.

3. **Partición del conjunto de datos**  
   Mediante `make_train_test_split`, el dataset se divide en tres subconjuntos:
   - `df_seed`: conjunto inicial para entrenar el modelo global;
   - `df_pool`: conjunto desde el que se simulan los clientes federados;
   - `df_test`: conjunto independiente para evaluación.

4. **Entrenamiento inicial del modelo global**  
   El modelo seleccionado para features (`linear` o `mlp`) se ajusta primero con `df_seed`, estableciendo una línea base antes de iniciar las rondas federadas.

5. **Simulación de rondas federadas**  
   En cada ronda:
   - `MobileFleetSimulator` selecciona clientes simulados y asigna muestras locales;
   - cada cliente realiza una actualización local;
   - las actualizaciones se combinan mediante la lógica de agregación correspondiente al modelo;
   - el modelo global actualizado se evalúa sobre `df_test`.

6. **Selección de checkpoints y persistencia de resultados**  
   Al finalizar la ejecución, el sistema guarda:
   - el modelo final;
   - el mejor modelo según `balanced_accuracy`;
   - el mejor modelo según `macro_f1`;
   - el histórico de métricas;
   - y los artefactos de evaluación generados por `reporting/reports.py`.

### B. Flujo del experimento image-based

En la rama basada en imagen, el flujo es análogo, pero cambiando la naturaleza de la entrada y del modelo:

1. **Construcción del índice de imágenes**  
   `ImageDataset` carga la metadata, verifica las rutas válidas y prepara el acceso a las muestras de imagen.

2. **Selección del modo de entrada visual**  
   Según la configuración, cada muestra puede construirse como:
   - imagen original;
   - imagen preprocesada;
   - imagen segmentada;
   - o imagen preprocesada con máscara.

3. **Partición en seed, pool y test**  
   La metadata de imágenes se divide en:
   - `df_seed`;
   - `df_pool`;
   - `df_test`.

   Además, el sistema garantiza que el conjunto seed contenga representación de todas las clases presentes en el dataset experimental.

4. **Entrenamiento inicial del modelo convolucional**  
   El modelo `cnn` se entrena primero con el conjunto seed y se evalúa sobre test para establecer una línea base.

5. **Rondas federadas sobre imagen**  
   En cada ronda:
   - se simulan clientes con subconjuntos locales de imágenes;
   - cada cliente actualiza localmente la CNN;
   - el servidor agrega los `state_dict` locales mediante media ponderada;
   - el nuevo modelo global se evalúa sobre test.

6. **Guardado de modelos e histórico**  
   Igual que en la rama feature-based, se guardan modelos, histórico y figuras del experimento.

### Estado actual de la ejecución

En este momento, ambos flujos son **operativos** dentro del marco de simulación federada offline:

- la rama **feature-based** se encuentra más madura y ofrece resultados experimentales más sólidos;
- la rama **image-based** ya funciona de extremo a extremo, aunque sigue en fase de ajuste y exploración.

---

## 📈 Evaluación y métricas

La evaluación del sistema se realiza sobre un conjunto de test separado tanto del entrenamiento inicial como de la simulación de clientes. En la versión actual del repositorio se emplean las siguientes métricas principales:

- **Accuracy**  
  Mide la proporción total de predicciones correctas.

- **Balanced Accuracy**  
  Resulta especialmente importante en HAM10000, ya que compensa el efecto del desbalanceo entre clases y ofrece una visión más justa del rendimiento por categoría.

- **Macro F1-score**  
  Resume el equilibrio entre precisión y exhaustividad asignando el mismo peso a cada clase, lo que la hace especialmente útil cuando existen clases minoritarias.

Además, el proyecto genera artefactos de evaluación complementarios:

- **classification report**, con precisión, recall y F1-score por clase;
- **matriz de confusión**, para analizar visualmente los errores de clasificación;
- **histórico de métricas por ronda**, para estudiar la evolución del modelo global a lo largo del entrenamiento federado.

En la rama basada en imagen, el reporting detallado por clases todavía está menos desarrollado que en la rama de características, aunque la infraestructura principal de evaluación ya está operativa.

---

## ⚖️ Justificación de ingeniería

La organización actual del proyecto responde a varias decisiones de diseño relevantes para el TFG:

1. **Separación entre rama feature-based y rama image-based**  
   El sistema soporta dos estrategias complementarias para abordar el problema: una basada en variables extraídas manualmente y otra basada en entrada directa de imagen.

2. **Desacoplamiento entre procesamiento visual y experimentación**  
   En la rama de features, el procesamiento de imagen se desacopla del entrenamiento federado mediante un dataset cacheado, reduciendo tiempos de ejecución y facilitando la experimentación.

3. **Diseño modular del software**  
   El código se divide en módulos de configuración, datos, visión, modelos, federación, reporting, servidor, edge y utilidades. Esto hace más sencillo mantener el proyecto y extenderlo en futuras fases del TFG.

4. **Simulación explícita de escenarios Non-IID**  
   El uso de `MobileFleetSimulator` permite estudiar cómo cambia el rendimiento cuando los clientes no comparten la misma distribución de clases.

5. **Privacidad como hipótesis de trabajo del sistema**  
   Aunque la implementación actual es una simulación offline, ya incorpora la posibilidad de perturbar actualizaciones locales con ruido gaussiano. Esto sirve como base experimental para discutir privacidad en entornos distribuidos.

6. **Preparación para evolución futura**  
   La existencia de módulos como `server/hospital_server.py` y `edge/mobile_app.py` permite distinguir desde ahora entre la simulación experimental actual y una futura arquitectura más cercana a un despliegue distribuido realista.

7. **Comparabilidad experimental**  
   Mantener una interfaz similar entre modelos y experimentos facilita comparar configuraciones distintas dentro de una misma base de software.

---

## 🚀 Ejecución

### Experimento basado en características

```bash
python experiments/run_federated_features.py
```

### Experimento basado en imagenes

```bash
python experiments/run_federated_images.py
```

Ambos scripts están pensados para poder lanzarse tanto desde terminal como desde entornos como Spyder, manteniendo un flujo de trabajo reproducible.

---

## 📊 Baselines y estado actual del proyecto

La siguiente tabla resume el papel de los principales enfoques implementados o utilizados como referencia dentro del proyecto.

| Enfoque | Tipo de entrada | Escenario | Estado en el proyecto | Papel experimental | Observaciones |
|---|---|---|---|---|---|
| **SVM centralizada (RBF)** | Features extraídas | Centralizado | Baseline externo de referencia | Referencia de techo clásico sobre features | Ofrece una referencia útil para estimar el potencial del pipeline de visión clásica, pero no encaja de forma natural en el esquema federado incremental implementado. |
| **Modelo lineal federado** | Features extraídas | Federado offline | Operativo y estable | Baseline federado principal | Es el modelo más maduro dentro del repositorio actual. Permite estudiar evolución por rondas, efecto Non-IID y agregación de actualizaciones locales. |
| **MLP federado** | Features extraídas | Federado offline | Operativo | Extensión del baseline feature-based | Introduce mayor capacidad expresiva que el modelo lineal, manteniendo compatibilidad con el marco federado actual. Sigue en fase de ajuste experimental. |
| **CNN federada básica** | Imagen directa | Federado offline | Operativa | Primera baseline image-based | Ya funciona de extremo a extremo dentro del pipeline federado, pero su rendimiento actual todavía está por debajo de la rama feature-based y requiere ajuste adicional. |
| **AlexNet / CNN más profunda** | Imagen directa | Centralizado o federado futuro | Referencia de cuaderno / línea futura | Extensión prevista | Existe como referencia experimental previa en cuadernos, pero todavía no está integrada como backend completo dentro de la arquitectura modular actual. |

### Lectura de esta comparación

- La rama **feature-based** es actualmente la más sólida del proyecto.
- La **SVM centralizada** sirve como referencia útil para comprobar que las características extraídas contienen señal discriminativa relevante.
- El **modelo lineal federado** actúa como baseline principal del sistema en su estado actual.
- La rama **image-based** ya es funcional, pero debe entenderse todavía como una línea en evolución y ajuste.
- En consecuencia, el repositorio combina una base experimental ya operativa con una hoja de ruta clara hacia modelos de imagen más potentes y una arquitectura federada más realista.

---

**Autor:** Paula Calvo  
**Tutor:** Fran J. Glez  
**Universidad:** [uc3m.es](https://www.uc3m.es)
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import joblib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


from scipy import ndimage as ndi
from skimage import color, exposure, filters, measure, morphology, segmentation, util
from skimage.feature import graycomatrix, graycoprops
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)
ZERO_FEATURES = np.zeros(11, dtype=np.float32)

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS REALES EN DRIVE
# ==========================================
# Montar Drive

# Rutas basadas en tu estructura de HAM10000
BASE_PATH = Path("/Users/fran/fran@ing.uc3m.es - Google Drive/Mi unidad/ham10000")
METADATA_PATH = BASE_PATH / "HAM10000_metadata.csv"
IMAGES_DIR = BASE_PATH / "dataset"
MODEL_ARTIFACT = BASE_PATH / Path("artifacts/global_model.joblib")

# Crear carpeta de salida para el modelo si no existe
MODEL_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. PIPELINE DE VISIÓN (Lógica ABCD + GLCM)
# ==========================================
class VisionPipeline:
    def __init__(self, structuring_radius: int = 5) -> None:
        self.structuring_element = morphology.disk(structuring_radius)

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        equalized = exposure.equalize_adapthist(image_rgb)
        closed = self._apply_per_channel(equalized, morphology.closing, self.structuring_element)
        filtered = self._apply_per_channel(closed, filters.median, self.structuring_element)
        return filtered

    def segment_lesion(self, image_rgb: np.ndarray) -> Optional[measure._regionprops.RegionProperties]:
        gray = color.rgb2gray(image_rgb)
        elevation = filters.sobel(gray)
        threshold = filters.threshold_isodata(gray)
        markers = np.zeros_like(gray, dtype=np.int32)
        markers[gray > threshold] = 1
        markers[gray < threshold] = 2
        
        mask = segmentation.watershed(elevation, markers)
        mask = ndi.binary_fill_holes(mask - 1)
        mask = morphology.remove_small_objects(mask, min_size=800)
        mask = segmentation.clear_border(mask)
        
        labeled = morphology.label(mask)
        props = measure.regionprops(labeled)
        if not props: return None
        
        # Selección robusta de la lesión
        areas = np.array([region.area for region in props])
        largest_idx = int(np.argmax(areas))
        return props[largest_idx]

    def extract_features(self, image_rgb: np.ndarray, lesion_region: Optional[measure._regionprops.RegionProperties]) -> np.ndarray:
        if lesion_region is None or lesion_region.area <= 0:
            return ZERO_FEATURES.copy()

        area = float(lesion_region.area)
        gray = color.rgb2gray(image_rgb)
        
        # A: Asimetría
        mask = lesion_region.image
        h_diff = np.count_nonzero(mask & ~np.fliplr(mask))
        v_diff = np.count_nonzero(mask & ~np.flipud(mask))
        asymmetry = float(0.5 * ((h_diff / area) + (v_diff / area)))
        
        # B: Borde (Compactness)
        compactness = float((lesion_region.perimeter ** 2) / (4 * np.pi * area))
        
        # C: Color
        row_s, col_s = lesion_region.slice
        roi = image_rgb[row_s, col_s]
        color_stats = [float(np.std(roi[:,:,i]) / (np.max(roi[:,:,i]) + 1e-6)) for i in range(3)]
        
        # D: Diámetro y Textura
        eq_diam = float(lesion_region.equivalent_diameter)
        glcm = graycomatrix(util.img_as_ubyte(gray), [1], [0, np.pi/4, np.pi/2], symmetric=True, normed=True)
        texture = [float(np.mean(graycoprops(glcm, p))) for p in ["correlation", "homogeneity", "energy", "contrast"]]
        
        return np.array([asymmetry, float(lesion_region.eccentricity), compactness, *color_stats, eq_diam, *texture], dtype=np.float32)

    def process_image_path(self, image_path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None: return ZERO_FEATURES.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preproc = self.preprocess(img_rgb)
        region = self.segment_lesion(preproc)
        return self.extract_features(preproc, region)

    @staticmethod
    def _apply_per_channel(image: np.ndarray, func: Any, kernel: np.ndarray) -> np.ndarray:
        channels = [func(image[:, :, c], kernel) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

# ==========================================
# 3. NODO HOSPITAL Y NODO MÓVIL
# ==========================================
@dataclass(slots=True)
class HospitalServer:
    metadata_path: Path
    images_dir: Path
    vision: VisionPipeline = field(default_factory=VisionPipeline)
    model: Pipeline = field(default_factory=lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    ]))

    def build_dataset(self, limit: int = 100) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.metadata_path).head(limit)
        X, y = [], []
        LOGGER.info(f"Procesando {len(df)} imágenes de Drive...")
        
        for row in df.itertuples():
            path = self.images_dir / f"{row.image_id}.jpg"
            if not path.exists(): continue
            vector = self.vision.process_image_path(path)
            if not np.allclose(vector, 0.0):
                X.append(vector)
                y.append(row.dx)
        
        return np.vstack(X).astype(np.float32), np.asarray(y)

    def train_and_save(self, X, y, path):
        self.model.fit(X, y)
        joblib.dump(self.model, path)
        LOGGER.info(f"Modelo global guardado en: {path}")

@dataclass(slots=True)
class MobileApp:
    model_path: Path
    vision: VisionPipeline = field(default_factory=VisionPipeline)
    model: Optional[Pipeline] = field(init=False, default=None)
    last_vector: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        self.model = joblib.load(self.model_path)

    def infer(self, image_path: Path) -> str:
        features = self.vision.process_image_path(image_path)
        self.last_vector = features
        return str(self.model.predict(features.reshape(1, -1))[0])

    def get_federated_packet(self, label: str) -> Dict[str, Any]:
        noise = np.random.normal(0.0, 0.01, size=self.last_vector.shape)
        return {"features": (self.last_vector + noise).astype(np.float32), "label": label}

# ==========================================
# 4. EJECUCIÓN DEL FLUJO
# ==========================================
if __name__ == "__main__":
    try:
        # A. Entrenamiento en el Servidor (Hospital)
        server = HospitalServer(METADATA_PATH, IMAGES_DIR)
        X, y = server.build_dataset(limit=200) # Prueba con 200 imágenes
        server.train_and_save(X, y, MODEL_ARTIFACT)

        # B. Inferencia en el Borde (App Móvil)
        # Usamos la primera imagen disponible para el test
        test_img_id = pd.read_csv(METADATA_PATH).iloc[0]['image_id']
        test_path = IMAGES_DIR / f"{test_img_id}.jpg"
        
        if test_path.exists():
            app = MobileApp(MODEL_ARTIFACT)
            pred = app.infer(test_path)
            LOGGER.info(f"Resultado Inferencia Local: {pred}")

            # C. Envío de paquete federado (Privacidad Diferencial)
            packet = app.get_federated_packet(pred)
            LOGGER.info("Paquete federado generado y anonimizado con éxito.")
            
    except Exception as e:
        LOGGER.error(f"Error en la ejecución: {e}")
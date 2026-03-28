from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, exposure, filters, measure, morphology, segmentation, util
from skimage.feature import graycomatrix, graycoprops
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger(__name__)
ZERO_FEATURES = np.zeros(11, dtype=np.float32)


# -----------------------------
# Image processing
# -----------------------------

class VisionPipeline:
    """Preprocesses dermoscopic images, segments lesions and extracts features."""

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
        if not props:
            return None

        target_label = self._select_target_label(props)
        final_mask = labeled == target_label
        final_labeled = morphology.label(final_mask)
        final_props = measure.regionprops(final_labeled)
        return final_props[0] if final_props else None

    def extract_features(
        self,
        image_rgb: np.ndarray,
        lesion_region: Optional[measure._regionprops.RegionProperties],
    ) -> np.ndarray:
        if lesion_region is None:
            return ZERO_FEATURES.copy()

        area = float(lesion_region.area)
        if area <= 0:
            return ZERO_FEATURES.copy()

        gray = color.rgb2gray(image_rgb)
        lesion_mask = lesion_region.image

        asymmetry = self._compute_asymmetry(lesion_mask, area)
        eccentricity = float(lesion_region.eccentricity)
        compactness = float((lesion_region.perimeter ** 2) / (4 * np.pi * area)) if area > 0 else 0.0
        color_stats = self._compute_color_features(image_rgb, lesion_region)
        equivalent_diameter = float(lesion_region.equivalent_diameter)
        texture_stats = self._compute_texture_features(gray)

        features = np.array(
            [asymmetry, eccentricity, compactness, *color_stats, equivalent_diameter, *texture_stats],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def process_image_path(self, image_path: str | Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            LOGGER.warning("Could not read image: %s", image_path)
            return ZERO_FEATURES.copy()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = self.preprocess(image_rgb)
        lesion_region = self.segment_lesion(preprocessed)
        return self.extract_features(preprocessed, lesion_region)

    @staticmethod
    def _apply_per_channel(image: np.ndarray, func: Any, kernel: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return func(image, kernel)
        channels = [func(image[:, :, c], kernel) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

    @staticmethod
    def _select_target_label(props: list[measure._regionprops.RegionProperties]) -> int:
        areas = np.array([region.area for region in props], dtype=np.float32)
        extents = np.array([region.extent for region in props], dtype=np.float32)
        largest_idx = int(np.argmax(areas))

        if areas[largest_idx] >= 1200:
            return props[largest_idx].label

        good_extent = np.where(extents > 0.50)[0]
        if good_extent.size:
            best_idx = int(good_extent[np.argmax(areas[good_extent])])
            return props[best_idx].label

        return props[largest_idx].label

    @staticmethod
    def _compute_asymmetry(mask: np.ndarray, area: float) -> float:
        horizontal_diff = np.count_nonzero(mask & ~np.fliplr(mask))
        vertical_diff = np.count_nonzero(mask & ~np.flipud(mask))
        return float(0.5 * ((horizontal_diff / area) + (vertical_diff / area)))

    @staticmethod
    def _compute_color_features(
        image_rgb: np.ndarray,
        lesion_region: measure._regionprops.RegionProperties,
    ) -> tuple[float, float, float]:
        row_slice, col_slice = lesion_region.slice
        roi = image_rgb[row_slice, col_slice]
        if roi.size == 0:
            return 0.0, 0.0, 0.0

        stats: list[float] = []
        for channel in range(3):
            channel_data = roi[:, :, channel]
            max_val = float(np.max(channel_data))
            stats.append(float(np.std(channel_data) / max_val) if max_val > 0 else 0.0)
        return stats[0], stats[1], stats[2]

    @staticmethod
    def _compute_texture_features(gray_image: np.ndarray) -> tuple[float, float, float, float]:
        glcm = graycomatrix(
            image=util.img_as_ubyte(gray_image),
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            symmetric=True,
            normed=True,
        )
        correlation = float(np.mean(graycoprops(glcm, prop="correlation")))
        homogeneity = float(np.mean(graycoprops(glcm, prop="homogeneity")))
        energy = float(np.mean(graycoprops(glcm, prop="energy")))
        contrast = float(np.mean(graycoprops(glcm, prop="contrast")))
        return correlation, homogeneity, energy, contrast


# -----------------------------
# Model training and inference
# -----------------------------

@dataclass(slots=True)
class HospitalServer:
    metadata_path: Path
    images_dir: Path
    vision: VisionPipeline = field(default_factory=VisionPipeline)
    model: Pipeline = field(
        default_factory=lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
            ]
        )
    )

    def build_dataset(self, image_ids: Optional[Iterable[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        metadata = pd.read_csv(self.metadata_path)
        if image_ids is not None:
            image_id_set = set(image_ids)
            metadata = metadata[metadata["image_id"].isin(image_id_set)]

        features: list[np.ndarray] = []
        labels: list[str] = []

        for row in metadata.itertuples(index=False):
            image_path = self.images_dir / f"{row.image_id}.jpg"
            if not image_path.exists():
                LOGGER.debug("Missing image: %s", image_path)
                continue

            vector = self.vision.process_image_path(image_path)
            if np.allclose(vector, 0.0):
                continue

            features.append(vector)
            labels.append(row.dx)

        if not features:
            return np.empty((0, ZERO_FEATURES.size), dtype=np.float32), np.empty((0,), dtype=object)

        return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=object)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data is empty.")
        self.model.fit(X, y)

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

    def receive_federated_updates(self, packets: list[dict[str, Any]]) -> pd.DataFrame:
        """Store updates for later retraining. Real FL aggregation would go here."""
        rows = []
        for packet in packets:
            rows.append(
                {
                    **{f"f_{i}": float(value) for i, value in enumerate(packet["features"])},
                    "label": packet["label"],
                }
            )
        return pd.DataFrame(rows)


@dataclass(slots=True)
class MobileApp:
    model_path: Path
    vision: VisionPipeline = field(default_factory=VisionPipeline)
    differential_privacy_std: float = 0.01
    model: Optional[Pipeline] = field(init=False, default=None)
    last_vector: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model = joblib.load(self.model_path)

    def infer(self, image_path: str | Path) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        features = self.vision.process_image_path(image_path)
        if np.allclose(features, 0.0):
            raise ValueError(f"No valid features could be extracted from: {image_path}")

        self.last_vector = features
        return str(self.model.predict(features.reshape(1, -1))[0])

    def get_federated_packet(self, confirmed_label: str, seed: Optional[int] = None) -> Dict[str, Any]:
        if self.last_vector is None:
            raise RuntimeError("Run infer() before requesting a federated packet.")

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.differential_privacy_std, size=self.last_vector.shape)
        return {
            "features": (self.last_vector + noise).astype(np.float32),
            "label": confirmed_label,
        }


# -----------------------------
# Example usage
# -----------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    metadata_path = Path("data/HAM10000_metadata.csv")
    images_dir = Path("data/images")
    model_path = Path("artifacts/global_model.joblib")
    sample_image = Path("data/sample.jpg")

    if not metadata_path.exists() or not images_dir.exists():
        LOGGER.info("Update metadata_path and images_dir before running training.")
        return

    server = HospitalServer(metadata_path=metadata_path, images_dir=images_dir)
    X_train, y_train = server.build_dataset()
    server.train(X_train, y_train)
    server.save(model_path)

    if sample_image.exists():
        app = MobileApp(model_path=model_path)
        prediction = app.infer(sample_image)
        LOGGER.info("Prediction: %s", prediction)

        packet = app.get_federated_packet(confirmed_label=prediction, seed=42)
        updates_df = server.receive_federated_updates([packet])
        LOGGER.info("Received %d federated update(s).", len(updates_df))


if __name__ == "__main__":
    main()

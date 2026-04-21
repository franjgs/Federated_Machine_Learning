"""
Lógica de la aplicación móvil.

Este módulo representa el nodo Edge:
- carga el modelo global desplegado por el hospital,
- procesa imágenes localmente,
- hace inferencia,
- y construye el paquete federado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from federated_skin.models.global_model import FederatedGlobalModel
from federated_skin.vision.pipeline import VisionPipeline


class MobileApp:
    """
    Nodo móvil del sistema.

    Parameters
    ----------
    model_path : Path
        Ruta al modelo global desplegado.
    logger : object | None
        Logger opcional.
    """

    def __init__(self, model_path: Path, logger: object | None = None) -> None:
        self.logger = logger
        self.model = FederatedGlobalModel.load(model_path)
        self.vision = VisionPipeline()
        self.last_vector: np.ndarray | None = None

    def infer(self, image_path: Path) -> str:
        """
        Realiza inferencia local sobre una imagen.

        Parameters
        ----------
        image_path : Path
            Ruta a la imagen a procesar.

        Returns
        -------
        str
            Clase predicha.
        """
        features = self.vision.process_image_path(image_path)

        if features is None or np.allclose(features, 0.0):
            if self.logger is not None:
                self.logger.warning(
                    "No se pudieron extraer características válidas de %s",
                    image_path,
                )
            return "No se pudo inferir"

        self.last_vector = features
        prediction = self.model.predict(features.reshape(1, -1))[0]
        return str(prediction)

    def get_federated_packet(self, confirmed_label: str, noise_std: float = 0.01) -> dict[str, Any] | None:
        """
        Genera el paquete federado a enviar al hospital.

        Parameters
        ----------
        confirmed_label : str
            Etiqueta confirmada por el profesional sanitario.
        noise_std : float
            Desviación típica del ruido gaussiano añadido para simular
            privacidad diferencial local.

        Returns
        -------
        dict[str, Any] | None
            Paquete con features perturbadas y etiqueta confirmada.
        """
        if self.last_vector is None:
            if self.logger is not None:
                self.logger.warning("No se ha ejecutado infer() antes de pedir el paquete federado.")
            return None

        noise = np.random.normal(0.0, noise_std, size=self.last_vector.shape)

        return {
            "features": (self.last_vector + noise).astype(np.float64),
            "label": confirmed_label,
        }
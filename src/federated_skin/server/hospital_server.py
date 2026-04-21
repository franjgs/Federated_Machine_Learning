"""
Lógica del servidor hospitalario.

Este módulo representa el nodo central del sistema:
- prepara el dataset inicial,
- entrena el modelo global,
- lo guarda para su despliegue,
- y recibe paquetes federados de los nodos móviles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from federated_skin.data.dataset_builder import FeatureDatasetBuilder
from federated_skin.data.splits import df_to_xy
from federated_skin.models.global_model import FederatedGlobalModel
from federated_skin.vision.pipeline import VisionPipeline


class HospitalServer:
    """
    Nodo hospitalario del sistema.

    Parameters
    ----------
    metadata_path : Path
        Ruta al fichero de metadata.
    images_dir : Path
        Directorio con imágenes dermatológicas.
    logger : object | None
        Logger opcional.
    """

    def __init__(
        self,
        metadata_path: Path,
        images_dir: Path,
        logger: object | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.images_dir = images_dir
        self.logger = logger
        self.vision = VisionPipeline()
        self.model = FederatedGlobalModel()

    def build_feature_dataset(self, cache_path: Path) -> pd.DataFrame:
        """
        Construye o carga el dataset tabular de características.

        Parameters
        ----------
        cache_path : Path
            Ruta de la caché parquet.

        Returns
        -------
        pd.DataFrame
            Dataset de features.
        """
        builder = FeatureDatasetBuilder(
            metadata_path=self.metadata_path,
            images_dir=self.images_dir,
            vision=self.vision,
            logger=self.logger,
        )
        return builder.build_or_load(cache_path)

    def train_initial_model(self, df_features: pd.DataFrame) -> None:
        """
        Entrena el modelo global inicial a partir de un DataFrame de features.

        Parameters
        ----------
        df_features : pd.DataFrame
            Dataset tabular con columnas f0, f1, ... y dx.
        """
        X, y = df_to_xy(df_features)
        if len(X) == 0:
            raise ValueError("No hay datos válidos para entrenar el modelo inicial.")

        self.model.fit_initial(X, y)

    def deploy_model(self, model_path: Path) -> None:
        """
        Guarda el modelo global para su despliegue.

        Parameters
        ----------
        model_path : Path
            Ruta donde se guardará el modelo.
        """
        self.model.save(model_path)

    def receive_federated_update(self, federated_packets: list[dict[str, Any]]) -> None:
        """
        Recibe paquetes federados enviados por nodos móviles.

        Esta versión es un placeholder conceptual. En el futuro aquí
        se podrá añadir la lógica real de agregación o reentrenamiento.

        Parameters
        ----------
        federated_packets : list[dict[str, Any]]
            Lista de paquetes con features y etiquetas confirmadas.
        """
        if self.logger is not None:
            self.logger.info("HospitalServer recibió %d paquetes federados.", len(federated_packets))

        for packet in federated_packets:
            if self.logger is not None:
                self.logger.info(
                    "Paquete federado | label=%s | n_features=%d",
                    packet["label"],
                    len(packet["features"]),
                )
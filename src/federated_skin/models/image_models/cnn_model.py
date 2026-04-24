"""Modelo federado CNN básico sobre imágenes dermatoscópicas."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_skin.models.base import FederatedModelBase


class _SimpleCNN(nn.Module):
    """
    CNN sencilla para clasificación de imágenes.

    Parameters
    ----------
    in_channels : int
        Número de canales de entrada.
    num_classes : int
        Número de clases de salida.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el forward de la red.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada de forma (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits de salida.
        """
        x = self.features(x)
        return self.classifier(x)


class FederatedCNNModel(FederatedModelBase):
    """
    Modelo CNN básico para clasificación de imágenes en un contexto federado.

    Parameters
    ----------
    random_state : int
        Semilla aleatoria para reproducibilidad.
    in_channels : int
        Número de canales de entrada de la imagen.
    learning_rate : float
        Tasa de aprendizaje.
    batch_size : int
        Tamaño de batch.
    """

    def __init__(
        self,
        random_state: int = 42,
        in_channels: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
    ) -> None:
        self.random_state = random_state
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.label_encoder = LabelEncoder()
        self.device = self._get_device()

        self.model: _SimpleCNN | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion = nn.CrossEntropyLoss()

        self.is_fitted = False

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def fit_initial(
        self,
        X: np.ndarray,
        y_str: Sequence[str],
        initial_epochs: int = 5,
    ) -> None:
        """
        Entrena el modelo global inicial usando el conjunto seed.

        Parameters
        ----------
        X : np.ndarray
            Imágenes de entrada con forma (n, h, w, c).
        y_str : Sequence[str]
            Etiquetas en formato texto.
        initial_epochs : int
            Número de épocas iniciales.
        """
        y = self.label_encoder.fit_transform(np.asarray(y_str))
        num_classes = len(self.label_encoder.classes_)

        self._build_model(num_classes=num_classes)
        train_loader = self._build_dataloader(X, y, shuffle=True)

        self.model.train()
        for _ in range(initial_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases en formato texto.

        Parameters
        ----------
        X : np.ndarray
            Imágenes de entrada con forma (n, h, w, c) o (h, w, c).

        Returns
        -------
        np.ndarray
            Predicciones en etiquetas originales.
        """
        self._check_fitted()

        x_tensor = self._to_tensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

        return self.label_encoder.inverse_transform(y_pred)

    def evaluate(self, X: np.ndarray, y_str: Sequence[str]) -> Dict[str, float]:
        """
        Evalúa el modelo sobre un conjunto de datos.

        Parameters
        ----------
        X : np.ndarray
            Imágenes de entrada.
        y_str : Sequence[str]
            Etiquetas reales en formato texto.

        Returns
        -------
        Dict[str, float]
            Diccionario con accuracy, balanced_accuracy y macro_f1.
        """
        self._check_fitted()

        y_true = self.label_encoder.transform(np.asarray(y_str))
        x_tensor = self._to_tensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def classification_details(
        self,
        X: np.ndarray,
        y_str: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Devuelve información detallada de clasificación.

        Parameters
        ----------
        X : np.ndarray
            Imágenes de entrada.
        y_str : Sequence[str]
            Etiquetas reales en formato texto.

        Returns
        -------
        Dict[str, Any]
            Diccionario con:
            - labels
            - classification_report
            - confusion_matrix
            - y_true
            - y_pred
        """
        self._check_fitted()

        y_true_str = np.asarray(y_str)
        y_pred_str = self.predict(X)
        labels = list(self.label_encoder.classes_)

        report_dict = classification_report(
            y_true_str,
            y_pred_str,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

        return {
            "labels": labels,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "y_true": y_true_str,
            "y_pred": y_pred_str,
        }

    def client_update(
        self,
        X_client: np.ndarray,
        y_client_str: Sequence[str],
        local_epochs: int = 1,
        noise_std: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Simula la actualización local de un cliente móvil.

        Parameters
        ----------
        X_client : np.ndarray
            Imágenes locales del cliente.
        y_client_str : Sequence[str]
            Etiquetas locales del cliente.
        local_epochs : int
            Número de épocas locales.
        noise_std : float
            Desviación típica del ruido gaussiano añadido a los pesos.

        Returns
        -------
        Dict[str, Any]
            Diccionario con:
            - state_dict
            - n_samples
        """
        self._check_fitted()

        y_client = self.label_encoder.transform(np.asarray(y_client_str))
        local_model = copy.deepcopy(self.model).to(self.device)
        local_optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.learning_rate,
        )

        train_loader = self._build_dataloader(X_client, y_client, shuffle=True)

        local_model.train()
        for _ in range(local_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                local_optimizer.zero_grad()
                logits = local_model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                local_optimizer.step()

        state_dict = {}
        for key, value in local_model.state_dict().items():
            tensor = value.detach().cpu().clone()

            if noise_std > 0.0 and torch.is_floating_point(tensor):
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=tensor.shape,
                )
                tensor = tensor + noise

            state_dict[key] = tensor

        return {
            "state_dict": state_dict,
            "n_samples": len(X_client),
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado serializable del modelo.

        Returns
        -------
        Dict[str, Any]
            Estado del modelo.
        """
        self._check_fitted()

        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        return {
            "state_dict": state_dict,
            "label_classes_": self.label_encoder.classes_.copy(),
            "in_channels": self.in_channels,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Reconstruye el modelo a partir de un estado guardado.

        Parameters
        ----------
        state : Dict[str, Any]
            Estado previamente generado con get_state().
        """
        self.label_encoder.classes_ = state["label_classes_"].copy()
        self.in_channels = state["in_channels"]
        self.learning_rate = state["learning_rate"]
        self.batch_size = state["batch_size"]

        num_classes = len(self.label_encoder.classes_)
        self._build_model(num_classes=num_classes)
        self.model.load_state_dict(state["state_dict"])

        self.is_fitted = True

    def save(self, path: Path) -> None:
        """
        Guarda el modelo en disco.

        Parameters
        ----------
        path : Path
            Ruta de salida.
        """
        payload = {
            "state": self.get_state(),
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "FederatedCNNModel":
        """
        Carga un modelo previamente guardado.

        Parameters
        ----------
        path : Path
            Ruta del fichero guardado.

        Returns
        -------
        FederatedCNNModel
            Modelo reconstruido.
        """
        payload = joblib.load(path)
        state = payload["state"]

        model = cls(
            random_state=payload["random_state"],
            in_channels=state["in_channels"],
            learning_rate=state["learning_rate"],
            batch_size=state["batch_size"],
        )
        model.set_state(state)
        return model

    def _build_model(self, num_classes: int) -> None:
        """
        Construye la red y el optimizador.

        Parameters
        ----------
        num_classes : int
            Número de clases de salida.
        """
        self.model = _SimpleCNN(
            in_channels=self.in_channels,
            num_classes=num_classes,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _build_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        """
        Construye un DataLoader a partir de arrays NumPy.

        Parameters
        ----------
        X : np.ndarray
            Imágenes con forma (n, h, w, c).
        y : np.ndarray
            Etiquetas codificadas.
        shuffle : bool
            Si se mezclan las muestras.

        Returns
        -------
        DataLoader
            Cargador de datos.
        """
        x_tensor = self._to_tensor(X)
        y_tensor = torch.as_tensor(y, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """
        Convierte imágenes NHWC a tensor NCHW de PyTorch.

        Parameters
        ----------
        X : np.ndarray
            Imágenes con forma (n, h, w, c) o (h, w, c).

        Returns
        -------
        torch.Tensor
            Tensor float32 con forma (n, c, h, w).
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)

        X = np.transpose(X, (0, 3, 1, 2))
        return torch.as_tensor(X, dtype=torch.float32)

    def _get_device(self) -> torch.device:
        """
        Selecciona el dispositivo disponible.

        Returns
        -------
        torch.device
            Dispositivo de ejecución.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _check_fitted(self) -> None:
        """
        Comprueba que el modelo haya sido entrenado o restaurado.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo CNN todavía no está entrenado.")
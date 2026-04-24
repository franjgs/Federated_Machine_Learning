"""Factoría de modelos federados basados en imagen."""

from __future__ import annotations

from federated_skin.models.image_models.cnn_model import FederatedCNNModel


def build_image_model(
    model_type: str,
    random_state: int = 42,
    in_channels: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
):
    """
    Construye un modelo federado basado en imágenes.

    Parameters
    ----------
    model_type : str
        Tipo de modelo. Valores soportados:
        - "cnn"
    random_state : int
        Semilla aleatoria.
    in_channels : int
        Número de canales de entrada.
    learning_rate : float
        Tasa de aprendizaje.
    batch_size : int
        Tamaño de batch.

    Returns
    -------
    object
        Instancia del modelo correspondiente.

    Raises
    ------
    ValueError
        Si el tipo no está soportado.
    """
    if model_type == "cnn":
        return FederatedCNNModel(
            random_state=random_state,
            in_channels=in_channels,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    raise ValueError(f"Tipo de modelo de imagen no soportado: {model_type}")
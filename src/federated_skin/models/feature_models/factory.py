"""Factoría de modelos federados basados en características."""

from __future__ import annotations

from federated_skin.models.feature_models.linear_model import FederatedLinearModel
from federated_skin.models.feature_models.mlp_model import FederatedMLPModel


def build_feature_model(
    model_type: str,
    random_state: int = 42,
):
    """
    Construye un modelo federado basado en vectores de características.

    Parameters
    ----------
    model_type : str
        Tipo de modelo. Valores soportados:
        - "linear"
        - "mlp"
    random_state : int
        Semilla aleatoria para reproducibilidad.

    Returns
    -------
    object
        Instancia del modelo federado correspondiente.

    Raises
    ------
    ValueError
        Si el tipo de modelo no está soportado.
    """
    if model_type == "linear":
        return FederatedLinearModel(random_state=random_state)

    if model_type == "mlp":
        return FederatedMLPModel(random_state=random_state)

    raise ValueError(f"Tipo de modelo de features no soportado: {model_type}")
"""Factoría de funciones de agregación para aprendizaje federado."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from federated_skin.federation.aggregation.linear import aggregate_client_states
from federated_skin.federation.aggregation.mlp import aggregate_mlp_client_states


FeatureAggregator = Callable[[Any, list[dict[str, Any]], float], None]


def build_feature_aggregator(model_type: str) -> FeatureAggregator:
    """
    Devuelve una función de agregación adecuada para el tipo de modelo.

    Parameters
    ----------
    model_type : str
        Tipo de modelo de features. Valores soportados:
        - "linear"
        - "mlp"

    Returns
    -------
    FeatureAggregator
        Función que actualiza el modelo global a partir de los estados de cliente.

    Raises
    ------
    ValueError
        Si el tipo de modelo no está soportado.
    """
    if model_type == "linear":
        return _aggregate_linear_model

    if model_type == "mlp":
        return _aggregate_mlp_model

    raise ValueError(f"Tipo de modelo de features no soportado: {model_type}")


def _aggregate_linear_model(
    global_model: Any,
    client_states: list[dict[str, Any]],
    server_lr: float,
) -> None:
    """
    Agrega estados de clientes en un modelo lineal.

    Parameters
    ----------
    global_model : Any
        Modelo global lineal.
    client_states : list[dict[str, Any]]
        Estados locales devueltos por los clientes.
    server_lr : float
        Peso de mezcla del servidor.
    """
    new_coef, new_intercept = aggregate_client_states(
        global_coef=global_model.model.coef_,
        global_intercept=global_model.model.intercept_,
        client_states=client_states,
        server_lr=server_lr,
    )

    global_model.model.coef_ = new_coef
    global_model.model.intercept_ = new_intercept


def _aggregate_mlp_model(
    global_model: Any,
    client_states: list[dict[str, Any]],
    server_lr: float,
) -> None:
    """
    Agrega estados de clientes en un modelo MLP.

    Parameters
    ----------
    global_model : Any
        Modelo global MLP.
    client_states : list[dict[str, Any]]
        Estados locales devueltos por los clientes.
    server_lr : float
        Peso de mezcla del servidor.
    """
    new_coefs, new_intercepts = aggregate_mlp_client_states(
        global_coefs=global_model.model.coefs_,
        global_intercepts=global_model.model.intercepts_,
        client_states=client_states,
        server_lr=server_lr,
    )

    global_model.model.coefs_ = new_coefs
    global_model.model.intercepts_ = new_intercepts
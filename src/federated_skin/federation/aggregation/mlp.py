"""Funciones de agregación para modelos MLP en aprendizaje federado."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_mlp_client_states(
    global_coefs: list[np.ndarray],
    global_intercepts: list[np.ndarray],
    client_states: list[dict[str, Any]],
    server_lr: float = 0.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Agrega las actualizaciones locales de varios clientes para un modelo MLP.

    Se calcula una media ponderada por número de muestras y después
    se mezcla con el estado global anterior usando server_lr.

    Parameters
    ----------
    global_coefs : list[np.ndarray]
        Lista de matrices de pesos actuales del modelo global.
    global_intercepts : list[np.ndarray]
        Lista de vectores de sesgo actuales del modelo global.
    client_states : list[dict[str, Any]]
        Lista de estados locales de clientes. Cada estado debe contener:
        - "coefs_"
        - "intercepts_"
        - "n_samples"
    server_lr : float
        Peso de la nueva actualización agregada frente al modelo global previo.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Nuevos pesos y sesgos del modelo global.
    """
    if not client_states:
        return global_coefs, global_intercepts

    total_samples = sum(state["n_samples"] for state in client_states)
    if total_samples == 0:
        return global_coefs, global_intercepts

    avg_coefs = [
        np.zeros_like(layer, dtype=np.float64)
        for layer in global_coefs
    ]
    avg_intercepts = [
        np.zeros_like(layer, dtype=np.float64)
        for layer in global_intercepts
    ]

    for state in client_states:
        weight = state["n_samples"] / total_samples

        for layer_idx, layer in enumerate(state["coefs_"]):
            avg_coefs[layer_idx] += weight * layer

        for layer_idx, layer in enumerate(state["intercepts_"]):
            avg_intercepts[layer_idx] += weight * layer

    new_coefs = []
    new_intercepts = []

    for global_layer, avg_layer in zip(global_coefs, avg_coefs):
        new_layer = (1.0 - server_lr) * global_layer + server_lr * avg_layer
        new_coefs.append(new_layer)

    for global_layer, avg_layer in zip(global_intercepts, avg_intercepts):
        new_layer = (1.0 - server_lr) * global_layer + server_lr * avg_layer
        new_intercepts.append(new_layer)

    return new_coefs, new_intercepts
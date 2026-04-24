"""Funciones de agregación para redes neuronales en aprendizaje federado."""

from __future__ import annotations

from typing import Any

import torch


def aggregate_neural_client_states(
    global_state_dict: dict[str, torch.Tensor],
    client_states: list[dict[str, Any]],
    server_lr: float = 0.3,
) -> dict[str, torch.Tensor]:
    """
    Agrega las actualizaciones locales de varios clientes para una red neuronal.

    Se calcula una media ponderada por número de muestras y después
    se mezcla con el estado global anterior usando server_lr.

    Parameters
    ----------
    global_state_dict : dict[str, torch.Tensor]
        Estado global actual del modelo.
    client_states : list[dict[str, Any]]
        Lista de estados locales de clientes. Cada estado debe contener:
        - "state_dict"
        - "n_samples"
    server_lr : float
        Peso de la nueva actualización agregada frente al modelo global previo.

    Returns
    -------
    dict[str, torch.Tensor]
        Nuevo state_dict global agregado.
    """
    if not client_states:
        return {
            key: value.detach().cpu().clone()
            for key, value in global_state_dict.items()
        }

    total_samples = sum(state["n_samples"] for state in client_states)
    if total_samples == 0:
        return {
            key: value.detach().cpu().clone()
            for key, value in global_state_dict.items()
        }

    avg_state_dict = {
        key: torch.zeros_like(value, dtype=value.dtype)
        for key, value in global_state_dict.items()
    }

    for state in client_states:
        weight = state["n_samples"] / total_samples
        client_state_dict = state["state_dict"]

        for key in avg_state_dict:
            avg_state_dict[key] += client_state_dict[key] * weight

    new_state_dict = {}
    for key, global_tensor in global_state_dict.items():
        if torch.is_floating_point(global_tensor):
            new_tensor = (1.0 - server_lr) * global_tensor + server_lr * avg_state_dict[key]
        else:
            # Para tensores no flotantes (si aparecieran), se conserva el valor global.
            new_tensor = global_tensor.detach().cpu().clone()

        new_state_dict[key] = new_tensor

    return new_state_dict
"""
Utilidades sencillas para logging.
"""

from __future__ import annotations


def log_info(logger: object | None, message: str, *args: object) -> None:
    """
    Escribe un mensaje informativo si se ha proporcionado un logger.

    Parameters
    ----------
    logger : object | None
        Logger opcional.
    message : str
        Mensaje de logging.
    *args : object
        Argumentos para formatear el mensaje.
    """
    if logger is not None:
        logger.info(message, *args)
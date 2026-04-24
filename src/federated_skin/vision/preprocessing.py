"""Preprocesado básico de imágenes dermatoscópicas."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """
    Carga una imagen desde disco en formato RGB.

    Parameters
    ----------
    image_path : Path
        Ruta de la imagen.

    Returns
    -------
    np.ndarray
        Imagen en formato RGB.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    Redimensiona una imagen a tamaño cuadrado fijo.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada.
    image_size : int
        Tamaño de salida.

    Returns
    -------
    np.ndarray
        Imagen redimensionada.
    """
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen al rango [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada.

    Returns
    -------
    np.ndarray
        Imagen normalizada en float32.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    """
    Carga y preprocesa una imagen completa.

    Parameters
    ----------
    image_path : Path
        Ruta de la imagen.
    image_size : int
        Tamaño objetivo.

    Returns
    -------
    np.ndarray
        Imagen RGB preprocesada.
    """
    image = load_image(image_path)
    image = resize_image(image, image_size)
    image = normalize_image(image)
    return image
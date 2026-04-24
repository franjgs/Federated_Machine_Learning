"""Extracción de características manuales a partir de una lesión segmentada."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte


FEATURE_NAMES = [
    "AsymIdx",
    "Eccentricity",
    "CI",
    "StdR",
    "StdG",
    "StdB",
    "Diameter",
    "Correlation",
    "Homogeneity",
    "Energy",
    "Contrast",
]


def extract_handcrafted_features(
    image: np.ndarray,
    lesion_region: Any,
) -> np.ndarray:
    """
    Extrae un vector de características manuales a partir de una imagen y de
    la región segmentada asociada a la lesión.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.
    lesion_region : Any
        Región de lesión, típicamente un objeto devuelto por regionprops.

    Returns
    -------
    np.ndarray
        Vector de características en el orden definido por FEATURE_NAMES.
    """
    asymm_idx, ecc = _extract_asymmetry_features(lesion_region)
    compact_index = _extract_border_features(lesion_region)
    c_r, c_g, c_b = _extract_color_features(image, lesion_region)
    eq_diameter = _extract_diameter_features(lesion_region)
    correlation, homogeneity, energy, contrast = _extract_texture_features(image)

    return np.asarray(
        [
            asymm_idx,
            ecc,
            compact_index,
            c_r,
            c_g,
            c_b,
            eq_diameter,
            correlation,
            homogeneity,
            energy,
            contrast,
        ],
        dtype=np.float64,
    )


def _extract_asymmetry_features(lesion_region: Any) -> tuple[float, float]:
    """
    Extrae características de asimetría de la lesión.

    Parameters
    ----------
    lesion_region : Any
        Región segmentada de la lesión.

    Returns
    -------
    tuple[float, float]
        Índice de asimetría y excentricidad.
    """
    area_total = lesion_region.area
    img_mask = lesion_region.image

    horizontal_flip = np.fliplr(img_mask)
    diff_horizontal = img_mask * ~horizontal_flip

    vertical_flip = np.flipud(img_mask)
    diff_vertical = img_mask * ~vertical_flip

    diff_horizontal_area = np.count_nonzero(diff_horizontal)
    diff_vertical_area = np.count_nonzero(diff_vertical)

    asymm_idx = 0.5 * (
        (diff_horizontal_area / area_total) + (diff_vertical_area / area_total)
    )
    ecc = float(lesion_region.eccentricity)

    return float(asymm_idx), ecc


def _extract_border_features(lesion_region: Any) -> float:
    """
    Extrae una característica de irregularidad del borde.

    Parameters
    ----------
    lesion_region : Any
        Región segmentada de la lesión.

    Returns
    -------
    float
        Índice de compacidad.
    """
    area_total = lesion_region.area
    compact_index = (lesion_region.perimeter ** 2) / (4 * np.pi * area_total)
    return float(compact_index)


def _extract_color_features(
    image: np.ndarray,
    lesion_region: Any,
) -> tuple[float, float, float]:
    """
    Extrae características de variación cromática.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.
    lesion_region : Any
        Región segmentada de la lesión.

    Returns
    -------
    tuple[float, float, float]
        Desviaciones estándar normalizadas por canal RGB.
    """
    sliced = image[lesion_region.slice]
    lesion_r = sliced[:, :, 0]
    lesion_g = sliced[:, :, 1]
    lesion_b = sliced[:, :, 2]

    c_r = np.std(lesion_r) / max(np.max(lesion_r), 1e-8)
    c_g = np.std(lesion_g) / max(np.max(lesion_g), 1e-8)
    c_b = np.std(lesion_b) / max(np.max(lesion_b), 1e-8)

    return float(c_r), float(c_g), float(c_b)


def _extract_diameter_features(lesion_region: Any) -> float:
    """
    Extrae el diámetro equivalente de la lesión.

    Parameters
    ----------
    lesion_region : Any
        Región segmentada de la lesión.

    Returns
    -------
    float
        Diámetro equivalente.
    """
    return float(lesion_region.equivalent_diameter_area)


def _extract_texture_features(
    image: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Extrae características de textura basadas en GLCM.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.

    Returns
    -------
    tuple[float, float, float, float]
        Correlación, homogeneidad, energía y contraste.
    """
    gray_img = rgb2gray(image)

    glcm = graycomatrix(
        image=img_as_ubyte(gray_img),
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 2],
        symmetric=True,
        normed=True,
    )

    correlation = np.mean(graycoprops(glcm, prop="correlation"))
    homogeneity = np.mean(graycoprops(glcm, prop="homogeneity"))
    energy = np.mean(graycoprops(glcm, prop="energy"))
    contrast = np.mean(graycoprops(glcm, prop="contrast"))

    return (
        float(correlation),
        float(homogeneity),
        float(energy),
        float(contrast),
    )
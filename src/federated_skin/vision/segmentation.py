"""Segmentación basada en regiones para lesiones cutáneas."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation


def estimate_lesion_mask(image: np.ndarray) -> np.ndarray:
    """
    Estima una máscara binaria de la lesión a partir de una imagen RGB.

    El pipeline sigue una estrategia basada en:
    1. conversión a escala de grises,
    2. filtro de Sobel,
    3. marcadores con umbral ISODATA,
    4. watershed,
    5. postprocesado morfológico,
    6. selección de la región más probable de lesión.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.

    Returns
    -------
    np.ndarray
        Máscara binaria de la lesión con valores 0 y 1.
    """
    result = segment_lesion(image)
    return result["mask"]


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica una máscara binaria sobre una imagen RGB.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.
    mask : np.ndarray
        Máscara binaria.

    Returns
    -------
    np.ndarray
        Imagen segmentada.
    """
    if mask.ndim == 2:
        mask_3d = np.expand_dims(mask, axis=-1)
    else:
        mask_3d = mask

    return image * mask_3d


def segment_lesion(image: np.ndarray) -> dict[str, Any]:
    """
    Segmenta la lesión y devuelve información útil del proceso.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.

    Returns
    -------
    dict[str, Any]
        Diccionario con:
        - gray_image
        - elevation_map
        - markers
        - segmented_mask
        - border_cleared_mask
        - labeled_image
        - target_label
        - mask
        - lesion_region
        - segmented_image
    """
    gray_image = _to_grayscale(image)
    elevation_map = _compute_elevation_map(gray_image)
    markers = _build_markers(gray_image)
    segmented_mask = _run_watershed(elevation_map, markers)
    segmented_mask = _postprocess_segmentation(segmented_mask)

    border_cleared_mask = segmentation.clear_border(segmented_mask)
    labeled_image = morphology.label(border_cleared_mask)

    lesion_region, target_label, labeled_image = _select_target_region(
        labeled_image=labeled_image,
        segmented_mask=segmented_mask,
    )

    mask = (labeled_image == target_label).astype(np.uint8)
    segmented_image = apply_mask_to_image(image, mask)

    return {
        "gray_image": gray_image,
        "elevation_map": elevation_map,
        "markers": markers,
        "segmented_mask": segmented_mask,
        "border_cleared_mask": border_cleared_mask,
        "labeled_image": labeled_image,
        "target_label": target_label,
        "mask": mask,
        "lesion_region": lesion_region,
        "segmented_image": segmented_image,
    }


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a escala de grises.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB.

    Returns
    -------
    np.ndarray
        Imagen en escala de grises.
    """
    return color.rgb2gray(image)


def _compute_elevation_map(gray_image: np.ndarray) -> np.ndarray:
    """
    Calcula el mapa de elevación usando Sobel.

    Parameters
    ----------
    gray_image : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Mapa de elevación.
    """
    return filters.sobel(gray_image)


def _build_markers(gray_image: np.ndarray) -> np.ndarray:
    """
    Construye marcadores para watershed a partir de un umbral ISODATA.

    Parameters
    ----------
    gray_image : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Imagen de marcadores enteros.
    """
    markers = np.zeros_like(gray_image, dtype=np.int32)
    threshold = filters.threshold_isodata(gray_image)

    markers[gray_image > threshold] = 1
    markers[gray_image < threshold] = 2

    return markers


def _run_watershed(
    elevation_map: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """
    Ejecuta watershed sobre el mapa de elevación y los marcadores.

    Parameters
    ----------
    elevation_map : np.ndarray
        Mapa de elevación.
    markers : np.ndarray
        Marcadores para watershed.

    Returns
    -------
    np.ndarray
        Máscara binaria inicial de segmentación.
    """
    segmented = segmentation.watershed(elevation_map, markers)
    return segmented - 1


def _postprocess_segmentation(segmented_mask: np.ndarray) -> np.ndarray:
    """
    Mejora la segmentación inicial con operaciones morfológicas.

    Las operaciones aplicadas son:
    - rellenado de huecos,
    - eliminación de objetos pequeños.

    Parameters
    ----------
    segmented_mask : np.ndarray
        Máscara binaria inicial.

    Returns
    -------
    np.ndarray
        Máscara binaria postprocesada.
    """
    processed = ndi.binary_fill_holes(segmented_mask)
    # processed = morphology.remove_small_objects(processed, min_size=800)
    processed = morphology.remove_small_objects(processed, max_size=799)
    return processed.astype(bool)


def _select_target_region(
    labeled_image: np.ndarray,
    segmented_mask: np.ndarray,
) -> tuple[Any, int, np.ndarray]:
    """
    Selecciona la región que más probablemente corresponde a la lesión.

    La estrategia sigue dos etapas:
    1. si la mayor región tras clear_border tiene área suficiente, se elige;
    2. si no, se vuelve a la segmentación previa a clear_border y se aplica
       una heurística basada en área y extent.

    Parameters
    ----------
    labeled_image : np.ndarray
        Imagen etiquetada tras clear_border.
    segmented_mask : np.ndarray
        Máscara segmentada previa a clear_border.

    Returns
    -------
    tuple[Any, int, np.ndarray]
        Región seleccionada, etiqueta objetivo e imagen etiquetada final.
    """
    props = measure.regionprops(labeled_image)
    target_label = _choose_target_label_from_props(props)

    if target_label is not None:
        lesion_region = props[target_label - 1]
        labeled_image = _keep_only_target_label(labeled_image, target_label)
        return lesion_region, target_label, labeled_image

    labeled_image = morphology.label(segmented_mask)
    props = measure.regionprops(labeled_image)

    if not props:
        raise ValueError("No se ha podido identificar ninguna región de lesión válida.")

    target_label = _choose_target_label_with_extent_fallback(props)
    lesion_region = props[target_label - 1]
    labeled_image = _keep_only_target_label(labeled_image, target_label)

    return lesion_region, target_label, labeled_image


def _choose_target_label_from_props(props: list[Any]) -> int | None:
    """
    Elige la región mayor si existe y supera un área mínima.

    Parameters
    ----------
    props : list[Any]
        Lista de regiones devueltas por regionprops.

    Returns
    -------
    int | None
        Etiqueta elegida o None si no hay una región claramente válida.
    """
    if not props:
        return None

    areas = [region.area for region in props]
    max_region_idx = int(np.argmax(areas))

    if areas[max_region_idx] >= 1200:
        return int(props[max_region_idx].label)

    return None


def _choose_target_label_with_extent_fallback(props: list[Any]) -> int:
    """
    Selecciona la región objetivo usando una heurística basada en área y extent.

    Se examinan las regiones más grandes y se prioriza la primera que tenga
    extent mayor que 0.50. Si ninguna la cumple, se elige la mayor región.

    Parameters
    ----------
    props : list[Any]
        Lista de regiones devueltas por regionprops.

    Returns
    -------
    int
        Etiqueta de la región seleccionada.
    """
    areas = [region.area for region in props]
    extents = [region.extent for region in props]

    sorted_indices = list(np.argsort(areas))[::-1]
    candidate_indices = sorted_indices[:3]

    for idx in candidate_indices:
        if extents[idx] > 0.50:
            return int(props[idx].label)

    return int(props[candidate_indices[0]].label)


def _keep_only_target_label(
    labeled_image: np.ndarray,
    target_label: int,
) -> np.ndarray:
    """
    Conserva únicamente la región con la etiqueta objetivo.

    Parameters
    ----------
    labeled_image : np.ndarray
        Imagen etiquetada.
    target_label : int
        Etiqueta que se desea conservar.

    Returns
    -------
    np.ndarray
        Imagen etiquetada con el resto de regiones anuladas.
    """
    output = labeled_image.copy()
    output[output != target_label] = 0
    return output
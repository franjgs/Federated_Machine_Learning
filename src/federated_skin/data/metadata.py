"""Carga y validación de la metadata del dataset dermatológico."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_METADATA_COLUMNS = ("image_id", "dx")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Carga la metadata del dataset desde un fichero CSV.

    Parameters
    ----------
    metadata_path : Path
        Ruta al fichero CSV con la metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame con la metadata cargada.
    """
    df = pd.read_csv(metadata_path)
    validate_metadata_columns(df)
    return df


def validate_metadata_columns(df: pd.DataFrame) -> None:
    """
    Comprueba que la metadata contenga las columnas mínimas requeridas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de metadata.

    Raises
    ------
    ValueError
        Si falta alguna columna obligatoria.
    """
    missing_columns = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]

    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Faltan columnas obligatorias en la metadata: {missing_str}")


def attach_image_paths(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """
    Añade una columna con la ruta esperada de cada imagen.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de metadata.
    images_dir : Path
        Directorio donde se encuentran las imágenes.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con la columna 'image_path'.
    """
    df_out = df.copy()
    df_out["image_path"] = df_out["image_id"].apply(lambda image_id: images_dir / f"{image_id}.jpg")
    return df_out


def filter_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra únicamente las filas cuya imagen exista físicamente en disco.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'image_path'.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    if "image_path" not in df.columns:
        raise ValueError("Se requiere la columna 'image_path' para filtrar imágenes existentes.")

    return df[df["image_path"].apply(Path.exists)].reset_index(drop=True)
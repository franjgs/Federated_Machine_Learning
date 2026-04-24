"""Lógica principal del experimento federado basado en imágenes."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from federated_skin.config.settings import FederatedExperimentConfig
from federated_skin.data.image_dataset import ImageDataset
from federated_skin.data.splits import (
    ensure_all_dataset_classes_in_seed,
    make_train_test_split,
)
from federated_skin.federation.aggregation.nn import aggregate_neural_client_states
from federated_skin.federation.clients import MobileFleetSimulator
from federated_skin.models.base import FederatedModelBase
from federated_skin.models.image_models.factory import build_image_model
from federated_skin.utils.logging_utils import log_info


class FederatedImageExperimentResult:
    """
    Resultado final del experimento federado basado en imágenes.
    """

    def __init__(
        self,
        final_model: FederatedModelBase,
        best_balanced_model: FederatedModelBase,
        best_macro_f1_model: FederatedModelBase,
        history_df: pd.DataFrame,
        best_balanced_round: int,
        best_macro_f1_round: int,
        test_df: pd.DataFrame,
    ) -> None:
        self.final_model = final_model
        self.best_balanced_model = best_balanced_model
        self.best_macro_f1_model = best_macro_f1_model
        self.history_df = history_df
        self.best_balanced_round = best_balanced_round
        self.best_macro_f1_round = best_macro_f1_round
        self.test_df = test_df


class FederatedImageExperiment:
    """
    Orquestador del experimento federado basado en imágenes.

    Este experimento trabaja con imágenes como entrada directa al modelo.
    La señal concreta de entrada se controla con `image_input_mode`, por
    ejemplo:
    - "original"
    - "preprocessed"
    - "segmented"
    - "preprocessed_plus_mask"

    El tipo de modelo de imagen se selecciona mediante `image_model_type`.
    Actualmente se soporta:
    - "cnn"
    """

    def __init__(
        self,
        metadata_path: Path,
        images_dir: Path,
        config: FederatedExperimentConfig,
        logger: object | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.images_dir = images_dir
        self.config = config
        self.logger = logger

    def run(self) -> FederatedImageExperimentResult:
        """
        Ejecuta el experimento federado completo basado en imágenes.

        El flujo general es:
        1. construir el índice de imágenes,
        2. dividir la metadata en seed, pool móvil y test,
        3. garantizar cobertura de clases en seed,
        4. materializar tensores de imagen para seed y test,
        5. entrenar el modelo global inicial,
        6. simular rondas federadas con clientes móviles,
        7. agregar actualizaciones locales,
        8. evaluar sobre test fijo y aplicar early stopping.

        Returns
        -------
        FederatedImageExperimentResult
            Resultado final del experimento.
        """
        image_dataset = ImageDataset(
            metadata_path=self.metadata_path,
            images_dir=self.images_dir,
            image_size=self.config.image_size,
            image_input_mode=self.config.image_input_mode,
        )

        # Guardamos el índice original del ImageDataset para poder recuperar
        # correctamente las muestras después de hacer splits y reset_index.
        df_images = image_dataset.to_dataframe().copy()
        df_images["dataset_index"] = df_images.index

        df_seed, df_pool, df_test = make_train_test_split(
            df=df_images,
            seed_size=self.config.seed_size,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        df_seed, df_pool, df_test = ensure_all_dataset_classes_in_seed(
            df_seed=df_seed,
            df_pool=df_pool,
            df_test=df_test,
            label_col="dx",
            random_state=self.config.random_state,
        )

        log_info(self.logger, "Seed inicial (imagen): %d imágenes", len(df_seed))
        log_info(self.logger, "Pool móvil (imagen): %d imágenes", len(df_pool))
        log_info(self.logger, "Test fijo (imagen): %d imágenes", len(df_test))

        log_info(
            self.logger,
            "Clases presentes en seed (imagen): %s",
            sorted(df_seed["dx"].unique().tolist()),
        )
        log_info(
            self.logger,
            "Clases presentes en test (imagen): %s",
            sorted(df_test["dx"].unique().tolist()),
        )

        X_seed, y_seed = self._rows_to_xy(image_dataset, df_seed)
        X_test, y_test = self._rows_to_xy(image_dataset, df_test)

        in_channels = X_seed.shape[-1]

        global_model = build_image_model(
            model_type=self.config.image_model_type,
            random_state=self.config.random_state,
            in_channels=in_channels,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
        )
        global_model.fit_initial(
            X_seed,
            y_seed,
            initial_epochs=self.config.initial_epochs,
        )

        baseline = global_model.evaluate(X_test, y_test)

        log_info(
            self.logger,
            "Baseline | image_model=%s | image_mode=%s | acc=%.4f | bal_acc=%.4f | macro_f1=%.4f",
            self.config.image_model_type,
            self.config.image_input_mode,
            baseline["accuracy"],
            baseline["balanced_accuracy"],
            baseline["macro_f1"],
        )

        history_rows = [
            {
                "round": 0,
                "accuracy": baseline["accuracy"],
                "balanced_accuracy": baseline["balanced_accuracy"],
                "macro_f1": baseline["macro_f1"],
                "num_clients": 0,
                "num_images": len(df_seed),
            }
        ]

        best_bal_acc = baseline["balanced_accuracy"]
        best_macro_f1 = baseline["macro_f1"]
        best_balanced_round = 0
        best_macro_f1_round = 0

        best_state_bal = copy.deepcopy(global_model.get_state())
        best_state_f1 = copy.deepcopy(global_model.get_state())

        patience_counter = 0
        best_early_metric = baseline[self.config.early_stopping_metric]

        fleet = MobileFleetSimulator(
            df_pool=df_pool,
            random_state=self.config.random_state,
        )
        df_pool_reset = df_pool.reset_index(drop=True)

        for round_idx in range(1, self.config.rounds + 1):
            clients = fleet.sample_clients_without_replacement(
                num_clients=self.config.clients_per_round,
                images_per_client=self.config.local_images_per_client,
                class_bias=self.config.client_class_bias,
            )

            client_states = []
            total_images_this_round = 0

            for client in clients:
                client_df = df_pool_reset.iloc[client.row_indices].copy()

                if client_df.empty:
                    continue

                X_client, y_client = self._rows_to_xy(image_dataset, client_df)
                total_images_this_round += len(client_df)

                client_state = global_model.client_update(
                    X_client=X_client,
                    y_client_str=y_client,
                    local_epochs=self.config.local_epochs,
                    noise_std=self.config.privacy_noise_std,
                )
                client_states.append(client_state)

            self._aggregate_global_model(
                global_model=global_model,
                client_states=client_states,
            )

            metrics = global_model.evaluate(X_test, y_test)

            if metrics["balanced_accuracy"] > best_bal_acc:
                best_bal_acc = metrics["balanced_accuracy"]
                best_balanced_round = round_idx
                best_state_bal = copy.deepcopy(global_model.get_state())

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_macro_f1_round = round_idx
                best_state_f1 = copy.deepcopy(global_model.get_state())

            monitored_value = metrics[self.config.early_stopping_metric]

            if monitored_value > best_early_metric:
                best_early_metric = monitored_value
                patience_counter = 0
            else:
                patience_counter += 1

            history_rows.append(
                {
                    "round": round_idx,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "num_clients": len(client_states),
                    "num_images": total_images_this_round,
                }
            )

            log_info(
                self.logger,
                "Ronda %02d | image_model=%s | image_mode=%s | clientes=%d | imgs=%d | acc=%.4f | bal_acc=%.4f | macro_f1=%.4f",
                round_idx,
                self.config.image_model_type,
                self.config.image_input_mode,
                len(client_states),
                total_images_this_round,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
            )

            if patience_counter >= self.config.early_stopping_patience:
                log_info(
                    self.logger,
                    "Early stopping activado en ronda %02d (sin mejora en %s durante %d rondas).",
                    round_idx,
                    self.config.early_stopping_metric,
                    self.config.early_stopping_patience,
                )
                break

        history_df = pd.DataFrame(history_rows)

        best_balanced_model = build_image_model(
            model_type=self.config.image_model_type,
            random_state=self.config.random_state,
            in_channels=in_channels,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
        )
        best_balanced_model.set_state(best_state_bal)

        best_macro_f1_model = build_image_model(
            model_type=self.config.image_model_type,
            random_state=self.config.random_state,
            in_channels=in_channels,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
        )
        best_macro_f1_model.set_state(best_state_f1)

        return FederatedImageExperimentResult(
            final_model=global_model,
            best_balanced_model=best_balanced_model,
            best_macro_f1_model=best_macro_f1_model,
            history_df=history_df,
            best_balanced_round=best_balanced_round,
            best_macro_f1_round=best_macro_f1_round,
            test_df=df_test,
        )

    def _rows_to_xy(
        self,
        image_dataset: ImageDataset,
        df_rows: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Convierte un subconjunto de metadata en arrays de imagen y etiquetas.

        Parameters
        ----------
        image_dataset : ImageDataset
            Dataset de imágenes configurado.
        df_rows : pd.DataFrame
            Filas seleccionadas del índice de imágenes.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            Imágenes en formato NHWC y etiquetas en texto.
        """
        images = []
        labels = []

        for row in df_rows.itertuples():
            sample = image_dataset.get_sample(row.dataset_index)
            images.append(sample["image"])
            labels.append(sample["dx"])

        X = np.stack(images, axis=0).astype(np.float32)
        return X, labels

    def _aggregate_global_model(
        self,
        global_model: FederatedModelBase,
        client_states: list[dict],
    ) -> None:
        """
        Agrega las actualizaciones locales en el modelo global de imagen.

        Actualmente se utiliza agregación de redes neuronales basada en
        media ponderada de state_dict por número de muestras.

        Parameters
        ----------
        global_model : FederatedModelBase
            Modelo global actual.
        client_states : list[dict]
            Lista de estados locales devueltos por los clientes.
        """
        global_state = global_model.get_state()

        new_state_dict = aggregate_neural_client_states(
            global_state_dict=global_state["state_dict"],
            client_states=client_states,
            server_lr=self.config.server_lr,
        )

        new_state = dict(global_state)
        new_state["state_dict"] = new_state_dict

        global_model.set_state(new_state)
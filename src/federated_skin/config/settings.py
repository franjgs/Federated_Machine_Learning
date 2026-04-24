<<<<<<< HEAD
"""Configuración general del experimento federado.

Este módulo concentra los parámetros principales del proyecto para evitar
números "duros" repartidos por distintos ficheros.

Aquí se definen, por ejemplo:
- tamaño del conjunto inicial de entrenamiento,
- número de rondas federadas,
- número de clientes por ronda,
- tamaño local de cada cliente,
- parámetros del modelo sobre features,
- parámetros del modelo sobre imagen,
- y parámetros de agregación o early stopping.
"""


class FederatedExperimentConfig:
    """
    Parámetros principales del experimento de aprendizaje federado.

    Atributos
    ---------
    seed_size : int
        Número de muestras usadas para entrenar el modelo global inicial.

    test_size : float
        Proporción del dataset total reservada para evaluación final.

    rounds : int
        Número máximo de rondas federadas.

    clients_per_round : int
        Número de clientes móviles simulados en cada ronda.

    local_images_per_client : int
        Número de muestras que utiliza cada cliente en su actualización local.

    feature_model_type : str
        Tipo de modelo usado sobre el vector de características.
        Valores típicos:
        - "linear"
        - "mlp"

    image_model_type : str
        Tipo de modelo usado sobre imagen.
        Valores típicos:
        - "cnn"

    image_input_mode : str
        Tipo de entrada usado en el pipeline basado en imagen.
        Valores típicos:
        - "original"
        - "preprocessed"
        - "segmented"
        - "preprocessed_plus_mask"

    image_size : int
        Tamaño cuadrado al que se redimensionan las imágenes de entrada.

    learning_rate : float
        Tasa de aprendizaje de los modelos basados en imagen.

    batch_size : int
        Tamaño de batch usado en modelos basados en imagen.

    local_epochs : int
        Número de épocas locales que ejecuta cada cliente antes de enviar
        su actualización al servidor.
        
    initial_epochs : int
        Número de pasadas sobre el conjunto seed para entrenar el modelo
        global inicial antes de comenzar las rondas federadas.

    client_class_bias : float
        Grado de sesgo de clase en los clientes simulados.
        - Valores cercanos a 0.0 producen clientes más parecidos entre sí.
        - Valores altos producen clientes más no-IID.

    privacy_noise_std : float
        Desviación típica del ruido gaussiano añadido a la actualización local
        para simular privacidad diferencial local sencilla.

    server_lr : float
        Tasa de mezcla del servidor al agregar los modelos locales.
        Un valor menor suaviza la actualización global.

    early_stopping_metric : str
        Métrica monitorizada para detener el entrenamiento antes de tiempo.
        Valores típicos:
        - "balanced_accuracy"
        - "macro_f1"

    early_stopping_patience : int
        Número de rondas sin mejora permitidas antes de activar early stopping.

    random_state : int
        Semilla aleatoria para favorecer reproducibilidad.
    """

    def __init__(
        self,
        seed_size: int = 500,
        test_size: float = 0.20,
        rounds: int = 20,
        clients_per_round: int = 30,
        local_images_per_client: int = 12,
        feature_model_type: str = "linear",
        image_model_type: str = "cnn",
        image_input_mode: str = "preprocessed",
        image_size: int = 224,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        local_epochs: int = 1,
        initial_epochs: int = 20,
        client_class_bias: float = 0.60,
        privacy_noise_std: float = 0.0,
        server_lr: float = 0.30,
        early_stopping_metric: str = "balanced_accuracy",
        early_stopping_patience: int = 5,
        random_state: int = 42,
    ) -> None:
        
        self.seed_size = seed_size
        self.test_size = test_size
        self.rounds = rounds
        self.clients_per_round = clients_per_round
        self.local_images_per_client = local_images_per_client

        self.feature_model_type = feature_model_type
        self.image_model_type = image_model_type
        self.image_input_mode = image_input_mode
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.local_epochs = local_epochs
        self.initial_epochs = initial_epochs
        self.client_class_bias = client_class_bias
        self.privacy_noise_std = privacy_noise_std
        self.server_lr = server_lr
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

def get_default_config() -> FederatedExperimentConfig:
    """
    Devuelve la configuración por defecto del experimento.

    Esta función permite centralizar la configuración base en un único punto,
    facilitando su reutilización desde scripts de entrenamiento o depuración.

    Returns
    -------
    FederatedExperimentConfig
        Objeto con los parámetros por defecto del experimento.
    """
    return FederatedExperimentConfig()
=======
"""Configuración general del experimento federado.Este módulo concentra los parámetros principales del proyecto para evitarnúmeros "duros" repartidos por distintos ficheros.Aquí se definen, por ejemplo:- tamaño del conjunto inicial de entrenamiento,- número de rondas federadas,- número de clientes por ronda,- tamaño local de cada cliente,- y parámetros de agregación o early stopping."""class FederatedExperimentConfig:    """    Parámetros principales del experimento de aprendizaje federado.    Atributos    ---------    seed_size : int        Número de imágenes usadas para entrenar el modelo global inicial.    test_size : float        Proporción del dataset total reservada para evaluación final.    rounds : int        Número máximo de rondas federadas.    clients_per_round : int        Número de clientes móviles simulados en cada ronda.    local_images_per_client : int        Número de imágenes que utiliza cada cliente en su actualización local.    local_epochs : int        Número de épocas locales que ejecuta cada cliente antes de enviar        su actualización al servidor.            initial_epochs : int        Número de pasadas sobre el conjunto seed para entrenar el modelo        global inicial antes de comenzar las rondas federadas.    client_class_bias : float        Grado de sesgo de clase en los clientes simulados.        - Valores cercanos a 0.0 producen clientes más parecidos entre sí.        - Valores altos producen clientes más no-IID.    privacy_noise_std : float        Desviación típica del ruido gaussiano añadido a la actualización local        para simular privacidad diferencial local sencilla.    server_lr : float        Tasa de mezcla del servidor al agregar los modelos locales.        Un valor menor suaviza la actualización global.    early_stopping_metric : str        Métrica monitorizada para detener el entrenamiento antes de tiempo.        Valores típicos:        - "balanced_accuracy"        - "macro_f1"    early_stopping_patience : int        Número de rondas sin mejora permitidas antes de activar early stopping.    random_state : int        Semilla aleatoria para favorecer reproducibilidad.    """    def __init__(        self,        seed_size: int = 500,        test_size: float = 0.20,        rounds: int = 20,        clients_per_round: int = 30,        local_images_per_client: int = 12,        local_epochs: int = 1,        initial_epochs: int = 20,        client_class_bias: float = 0.60,        privacy_noise_std: float = 0.0,        server_lr: float = 0.30,        early_stopping_metric: str = "balanced_accuracy",        early_stopping_patience: int = 5,        random_state: int = 42,    ) -> None:                self.seed_size = seed_size        self.test_size = test_size        self.rounds = rounds        self.clients_per_round = clients_per_round        self.local_images_per_client = local_images_per_client        self.local_epochs = local_epochs        self.initial_epochs = initial_epochs        self.client_class_bias = client_class_bias        self.privacy_noise_std = privacy_noise_std        self.server_lr = server_lr        self.early_stopping_metric = early_stopping_metric        self.early_stopping_patience = early_stopping_patience        self.random_state = random_statedef get_default_config() -> FederatedExperimentConfig:    """    Devuelve la configuración por defecto del experimento.    Esta función permite centralizar la configuración base en un único punto,    facilitando su reutilización desde scripts de entrenamiento o depuración.    Returns    -------    FederatedExperimentConfig        Objeto con los parámetros por defecto del experimento.    """    return FederatedExperimentConfig()
>>>>>>> 23e1171 (Refactor architecture and simplify debuggable classes)

import gc
import torch

from nebula.core.aggregation.aggregator import Aggregator


class ProtoFedAvg(Aggregator):
    """
    Mix Prototype Model Aggregation

    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models = list(models.values())

        total_samples = float(sum(weight for _, weight in models))

        if total_samples == 0:
            raise ValueError("Total number of samples must be greater than zero.")

        # Obtener las claves de los parámetros de modelo desde el último modelo recibido
        last_model_params = models[-1][0]
        param_keys = [k for k in last_model_params.keys() if k != "protos"]

        # Inicializar acumuladores para los parámetros del modelo (excluyendo prototipos)
        accum_params = {key: torch.zeros_like(last_model_params[key], dtype=torch.float32) for key in param_keys}

        # Inicializar estructuras de datos para prototipos
        proto_accum = {}  # Acumular prototipos por clase
        proto_weights = {}  # Acumular pesos totales por clase

        with torch.no_grad():
            for model_parameters, weight in models:
                normalized_weight = weight / total_samples

                # Agregar parámetros estándar del modelo
                for key in param_keys:
                    if key in model_parameters:
                        accum_params[key].add_(model_parameters[key].to(accum_params[key].dtype), alpha=normalized_weight)
                    else:
                        # Si la clave no está en el modelo actual, podría ser debido al filtrado por send_logic
                        pass  # Puedes manejar esto si es necesario

                # Manejar prototipos si están presentes
                if "protos" in model_parameters:
                    client_protos = model_parameters["protos"]
                    for class_label, proto in client_protos.items():
                        if class_label not in proto_accum:
                            # Inicializar acumulador para esta clase
                            proto_accum[class_label] = proto.clone().detach() * normalized_weight
                            proto_weights[class_label] = normalized_weight
                        else:
                            # Acumular prototipos y pesos
                            proto_accum[class_label] += proto.clone().detach() * normalized_weight
                            proto_weights[class_label] += normalized_weight
                else:
                    # Si el modelo no contiene prototipos, continuamos
                    pass

        # Normalizar prototipos
        for class_label in proto_accum:
            proto_accum[class_label] /= proto_weights[class_label]

        # Fusionar los prototipos agregados de vuelta en los parámetros del modelo
        accum_params["protos"] = proto_accum

        # Marcamos con 'del' todas las variables intermedias
        del proto_accum, proto_weights, last_model_params, total_samples
        gc.collect()

        return accum_params

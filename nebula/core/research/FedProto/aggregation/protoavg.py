import gc
import logging
import torch
from nebula.core.aggregation.aggregator import Aggregator


class ProtoAvg(Aggregator):
    """
    Prototype Aggregation (FedProto) [Yue Tan et al., 2022]
    Paper: https://arxiv.org/abs/2105.00243
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, prototypes):
        super().run_aggregation(prototypes)

        # Convertir a una lista para poder iterar
        prototypes = list(prototypes.values())

        # Filtrar prototipos vacíos
        non_empty_prototypes = []
        for prototype, weight in prototypes:
            if prototype:
                non_empty_prototypes.append((prototype, weight))
            else:
                # Opcional: registrar que se omitió un prototipo vacío
                logging.debug("Se omitió un prototipo vacío.")

        if not non_empty_prototypes:
            # Manejar el caso en que todos los prototipos están vacíos
            logging.warning("Todos los prototipos están vacíos; no se puede realizar la agregación.")
            return dict()

        # Obtener el dispositivo del primer prototipo no vacío
        first_prototype = non_empty_prototypes[0][0]
        first_label = next(iter(first_prototype))
        device = first_prototype[first_label].device

        # Asegurarse de que 'weight' sea un escalar y mover prototipos al dispositivo correcto
        for i in range(len(non_empty_prototypes)):
            prototype, weight = non_empty_prototypes[i]
            # Convertir weight a escalar si es un tensor
            if isinstance(weight, torch.Tensor):
                weight = weight.to(device).item()
            else:
                weight = float(weight)
            # Mover los tensores del prototipo al dispositivo
            for label in prototype:
                prototype[label] = prototype[label].to(device)
            non_empty_prototypes[i] = (prototype, weight)

        # Sumar las muestras totales
        total_samples = sum(weight for _, weight in non_empty_prototypes)

        # Crear un prototipo acumulado en el dispositivo correcto
        accum = {}
        for label in first_prototype:
            accum[label] = torch.zeros_like(first_prototype[label]).to(device)

        # Agregar los modelos ponderados
        for prototype, weight in non_empty_prototypes:
            for label in prototype:
                if label not in accum:
                    accum[label] = torch.zeros_like(prototype[label]).to(device)
                accum[label] += prototype[label] * weight

        # Normalizar 'accum'
        for label in accum:
            accum[label] /= total_samples

        del prototypes, non_empty_prototypes, total_samples
        gc.collect()

        return accum

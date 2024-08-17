import logging

import torch
import numpy as np
from nebula.core.aggregation.aggregator import Aggregator
from nebula.core.aggregation.trimmedmean import TrimmedMean


class Bulyan(Aggregator):
    """
    Bulyan [El Mahdi El Mhamdi et al., 2018]
    Paper: The Hidden Vulnerability of Distributed Learning in Byzantium
    http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf
    Note: This implementation needs 5 participants to work properly. It would be possible to
    implement dynamically calculated selection set lengths and beta, but this only makes limited sense
    - Bulyan removes participant twice: once for the selection set and once for the trimmed mean
    TRM_BETA and KRUM_SELECTION_SET_LEN are hardcoded for now and should be changed to higher values
    when significantly more participants are used ...
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self.KRUM_SELECTION_SET_LEN = 4
        self.TRM_BETA = 1


    def run_aggregation(self, models):
        if len(models) == 0:
            logging.error("[Bulyan] Trying to aggregate models when there is no models")
            return None

        # Krum Step of Bulyan:
        # The implementation of the Krum Function is copied from krum.py [Author: Chao Feng].
        # This implementation was then modified to return a list of models ordered by their distance
        # instead of the single update with the best score to make it suitable for use in the Bulyan AGR

        models = list(models.values())

        # initialize zeroed model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        logging.info(
            "[Bulyan(Krum Step).aggregate] Aggregating models: num={}".format(
                len(models)
            )
        )

        # Create model distance list
        total_models = len(models)
        distance_list = [0 for i in range(0, total_models)]
        models_and_distances = []

        # Calculate the L2 Norm between xi and xj
        for i in range(0, total_models):
            m1, _ = models[i]
            for j in range(0, total_models):
                m2, _ = models[j]
                distance = 0
                if i == j:
                    distance = 0
                else:
                    for layer in m1:
                        l1 = m1[layer]
                        l2 = m2[layer]
                        distance += np.linalg.norm(l1 - l2)
                distance_list[i] += distance

            # Add the model and its distance to the dictionary containing all models and their distances
            models_and_distances.append((distance_list[i], models[i]))

        # Order the models by distance ascending -> potentially malicious models are at the end of the list
        models_and_distances.sort(key = lambda tup: tup[0])

        # remove the potentially malicious models
        if len(models_and_distances) <= self.KRUM_SELECTION_SET_LEN:
            logging.error(
                "[Bulyan(TRMstep)] Trying to aggregate models when there are less or equal models than the set length of the krum selection set ..."
            )
            return None
        else:
            for i in range(self.KRUM_SELECTION_SET_LEN):
                models_and_distances.pop()
        # calculate new global model using trimmedmean
        models = [x[1] for x in models_and_distances]
        TRM = TrimmedMean(config = self.config, beta = self.TRM_BETA)
        return TRM.run_aggregation(models)

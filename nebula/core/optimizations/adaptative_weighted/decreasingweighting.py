from nebula.core.optimizations.adaptative_weighted.weighting import Weighting


class DeacreasingWeighting(Weighting):
    def __init__(self, alpha_value=None, beta_value=None, lambda_value=None, limit=0.01, rounds=10):
        super(DeacreasingWeighting, self).__init__()
        self.alpha_init = alpha_value
        self.alpha_value = alpha_value
        self.beta_init = beta_value
        self.beta_value = beta_value
        self.lambda_init = lambda_value
        self.lambda_value = lambda_value
        self.limit = limit
        self.rounds = rounds

    def get_alpha(self, *args, **kwargs):
        return self.alpha_value

    def get_beta(self, *args, **kwargs):
        return self.beta_value

    def get_lambda(self, *args, **kwargs):
        return self.lambda_value

    def decreasing_step(self):
        if self.alpha_value is not None:
            self.alpha_value -= (2 * self.alpha_init) / self.rounds
        if self.beta_value is not None:
            self.beta_value -= (2 * self.beta_init) / self.rounds
        if self.lambda_value is not None:
            self.lambda_value -= (2 * self.lambda_init) / self.rounds

        if self.alpha_value < self.limit:
            self.alpha_value = 0
        if self.beta_value < self.limit:
            self.beta_value = 0
        if self.lambda_value < self.limit:
            self.lambda_value = 0

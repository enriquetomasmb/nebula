class Weighting:
    def __init__(self, alpha_value=None, beta_value=None, lambda_value=None):
        self.alpha_value = alpha_value
        self.beta_value = beta_value
        self.lambda_value = lambda_value

    def get_alpha(self, *args, **kwargs):
        return self.alpha_value

    def get_beta(self, *args, **kwargs):
        return self.beta_value

    def get_lambda(self, *args, **kwargs):
        return self.lambda_value

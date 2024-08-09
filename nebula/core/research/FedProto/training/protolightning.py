from nebula.core.training.lightning import Lightning


class ParameterProtosSettingError(Exception):
    """Custom exception for errors setting model parameters."""


class ProtoLightning(Lightning):
    """
    Learner with PyTorch Lightning.

    """

    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)

    def set_model_parameters(self, params, initialize=False):
        if initialize:
            self.model.load_state_dict(params)
            if hasattr(self.model, "set_protos"):
                self.model.set_protos(dict())
        else:
            if hasattr(self.model, "set_protos"):
                self.model.set_protos(params)
            else:
                try:
                    self.model.load_state_dict(params)
                except Exception as e:
                    raise ParameterProtosSettingError("Error setting parameters") from e

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())

            return self.model.state_dict()

        if hasattr(self.model, "get_protos"):
            return self.model.get_protos()

        if bytes:
            return self.serialize_model(self.model.state_dict())

        return self.model.state_dict()

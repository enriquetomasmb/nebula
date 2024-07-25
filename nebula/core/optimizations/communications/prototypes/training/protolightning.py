from nebula.core.training.lightning import Lightning


class ProtoLightning(Lightning):
    """
    Learner with PyTorch Lightning.

    """

    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)


    def set_model_parameters(self, params, initialize=False):
        if initialize:
            self.model.load_state_dict(params)
            if hasattr(self.model, 'set_protos'):
                self.model.set_protos(dict())
            return

        if hasattr(self.model, 'set_protos'):
            self.model.set_protos(params)
            return

        try:
            self.model.load_state_dict(params)
        except:
            raise Exception("Error setting parameters")

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())
            else:
                return self.model.state_dict()

        if bytes:
            if hasattr(self.model, 'get_protos'):
                return self.serialize_model(self.model.get_protos())
            return self.serialize_model(self.model.state_dict())
        else:
            if hasattr(self.model, 'get_protos'):
                return self.model.get_protos()
            return self.model.state_dict()


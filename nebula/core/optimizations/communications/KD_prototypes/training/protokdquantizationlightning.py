import logging
import traceback

from nebula.core.optimizations.communications.KD_prototypes.training.protoquantizationlightning import ProtoQuantizationLightning

class ProtoKDQuantizationLightning(ProtoQuantizationLightning):
    """
    Learner with PyTorch Lightning. Implements quantization of model parameters.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)


    def train(self):

        try:
            # activate anomaly detection
            # torch.autograd.set_detect_anomaly(True)
            if self.epochs > 0:
                # check if the model have a teacher model
                if hasattr(self.model, "teacher_model") and self.model.teacher_model is not None:
                    # check if the teacher model is using KD
                    if (hasattr(self.model.teacher_model, "set_student_model")):
                        # check if the student model was updated
                        if (hasattr(self.model, "model_updated_flag2") and hasattr(self.model, "model_updated_flag1")
                                and self.model.model_updated_flag2 and self.model.model_updated_flag1):
                            logging.info("[Learner] Mutual Distillation. Student model updated on teacher model.")
                            self.model.model_updated_flag2 = False
                            self.model.model_updated_flag1 = False
                            self.model.teacher_model.set_student_model(self.model)
                            if hasattr(self.model, "send_logic_step"):
                                logic = self.model.send_logic_step()
                                logging.info("[Learner] Logic step: {}".format(logic))
                        else:
                            logging.info("[Learner] Mutual Distillation. Student model not updated on teacher model.")
                            self.model.teacher_model.set_student_model(None)

                    else:
                        if (hasattr(self.model, "model_updated_flag2") and hasattr(self.model, "model_updated_flag1")
                                and self.model.model_updated_flag2 and self.model.model_updated_flag1):
                            if hasattr(self.model, "send_logic_step"):
                                logic = self.model.send_logic_step()
                                logging.info("[Learner] Logic step: {}".format(logic))

                    logging.info("[Learner] Training teacher model...")
                    # train the teacher model with Lightning
                    self.create_trainer()
                    self._trainer.fit(self.model.teacher_model, self.data)
                    self._trainer = None

                # train the student model with Lightning
                logging.info("[Learner] Training student model...")
                self.create_trainer()
                # torch.autograd.set_detect_anomaly(True)
                self._trainer.fit(self.model, self.data)
                self._trainer = None


        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            # Log full traceback
            logging.error(traceback.format_exc())
import logging
import traceback

from nebula.core.optimizations.communications.KD_prototypes.training.protoquantizationlightning import ProtoQuantizationLightning
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


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
        self._trainer = None

    async def train(self):

        try:
            # activate anomaly detection
            # torch.autograd.set_detect_anomaly(True)
            if self.epochs > 0:
                # check if the model have a teacher model
                if hasattr(self.model, "teacher_model") and self.model.teacher_model is not None:
                    # check if the teacher model is using KD
                    if hasattr(self.model.teacher_model, "set_student_model"):
                        # check if the student model was updated
                        if hasattr(self.model, "model_updated_flag") and self.model.model_updated_flag:
                            logging.info("[Learner] Mutual Distillation. Student model updated on teacher model.")
                            self.model.model_updated_flag = False
                            self.model.teacher_model.set_student_model(self.model)
                        else:
                            logging_training.info("[Learner] Mutual Distillation. Student model not updated on teacher model.")
                            self.model.teacher_model.set_student_model(None)

                    else:
                        if hasattr(self.model, "model_updated_flag") and self.model.model_updated_flag:
                            if hasattr(self.model, "send_logic_step"):
                                logic = self.model.send_logic_step()
                                logging_training.info(f"[Learner] Logic step: {logic}")

                    logging_training.info("[Learner] Training teacher model...")
                    # train the teacher model with Lightning
                    self.create_trainer()
                    self._trainer.fit(self.model.teacher_model, self.data)
                    self._trainer = None

                # train the student model with Lightning
                logging_training.info("[Learner] Training student model...")
                self.create_trainer()
                # torch.autograd.set_detect_anomaly(True)
                self._trainer.fit(self.model, self.data)
                self._trainer = None

        except RuntimeError as e:

            logging_training.error(f"Runtime issue with PyTorch Lightning: {e}")
            # Log full traceback
            logging_training.error(traceback.format_exc())

        except ValueError as e:
            logging_training.error(f"Value error encountered: {e}")
            # Log full traceback
            logging_training.error(traceback.format_exc())

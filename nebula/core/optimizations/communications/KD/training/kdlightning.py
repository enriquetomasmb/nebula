import logging
import traceback

from nebula.core.optimizations.communications.KD.training.quantizationlightning import QuantizationLightning
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


class KDLightning(QuantizationLightning):
    """
    Learner with PyTorch Lightning. Implements quantization of model parameters and training with knowledge distillation.


    Atributes:
        model: Model to train.
        data: Data to train the model.
        config: Configuration of the training.
        logger: Logger.
    """

    def __init__(self, model, data, config=None):
        super().__init__(model, data, config)
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
                        if (
                            hasattr(self.model, "model_updated_flag2")
                            and hasattr(self.model, "model_updated_flag1")
                            and self.model.model_updated_flag2
                            and self.model.model_updated_flag1
                        ):
                            logging_training.info("[Learner] Mutual Distillation. Student model updated on teacher model.")
                            self.model.model_updated_flag2 = False
                            self.model.model_updated_flag1 = False
                            self.model.teacher_model.set_student_model(self.model)
                            if hasattr(self.model, "send_logic_step"):
                                logic = self.model.send_logic_step()
                                logging_training.info(f"[Learner] Logic step: {logic}")
                        else:
                            logging.info("[Learner] Mutual Distillation. Student model not updated on teacher model.")
                            self.model.teacher_model.set_student_model(None)
                    else:
                        # check if we are in the case that does not use mutual distillation but using send logic
                        if (
                            hasattr(self.model, "model_updated_flag2")
                            and hasattr(self.model, "model_updated_flag1")
                            and self.model.model_updated_flag2
                            and self.model.model_updated_flag1
                        ):
                            if hasattr(self.model, "send_logic_step"):
                                logic = self.model.send_logic_step()
                                logging.info(f"[Learner] Logic step: {logic}")

                    # check if beta limit is reached, if so skip training teacher model
                    if hasattr(self.model, "beta") and hasattr(self.model, "limit_beta") and self.model.beta > self.model.limit_beta:
                        logging_training.info("[Learner] Training teacher model...")
                        # train the teacher model with Lightning
                        self.create_trainer()
                        self._trainer.fit(self.model.teacher_model, self.data)
                        self._trainer = None
                    else:
                        logging_training.info("[Learner] Beta limit reached. Skipping Training teacher model...")

                else:
                    # check if we are in the case that does not KD but using send logic
                    if (
                        hasattr(self.model, "model_updated_flag2")
                        and hasattr(self.model, "model_updated_flag1")
                        and self.model.model_updated_flag2
                        and self.model.model_updated_flag1
                    ):
                        if hasattr(self.model, "send_logic_step"):
                            logic = self.model.send_logic_step()
                            logging.info(f"[Learner] Logic step: {logic}")

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

    def on_learning_cycle_end(self):
        self._logger.log_data({"Round": self.round})
        self._logger.log_data({"Beta": self.model.beta}, step=self.logger.global_step)
        # self.reporter.enqueue_data("Round", self.round)

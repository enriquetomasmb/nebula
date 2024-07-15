:py:mod:`nebula.addons.trustworthiness.calculation`
===================================================

.. py:module:: nebula.addons.trustworthiness.calculation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   nebula.addons.trustworthiness.calculation.get_mapped_score
   nebula.addons.trustworthiness.calculation.get_normalized_scores
   nebula.addons.trustworthiness.calculation.get_range_score
   nebula.addons.trustworthiness.calculation.get_map_value_score
   nebula.addons.trustworthiness.calculation.get_true_score
   nebula.addons.trustworthiness.calculation.get_scaled_score
   nebula.addons.trustworthiness.calculation.get_value
   nebula.addons.trustworthiness.calculation.check_properties
   nebula.addons.trustworthiness.calculation.get_cv
   nebula.addons.trustworthiness.calculation.get_global_privacy_risk
   nebula.addons.trustworthiness.calculation.get_elapsed_time
   nebula.addons.trustworthiness.calculation.get_bytes_models
   nebula.addons.trustworthiness.calculation.get_bytes_sent_recv
   nebula.addons.trustworthiness.calculation.get_avg_loss_accuracy
   nebula.addons.trustworthiness.calculation.get_feature_importance_cv
   nebula.addons.trustworthiness.calculation.get_clever_score
   nebula.addons.trustworthiness.calculation.stop_emissions_tracking_and_save



Attributes
~~~~~~~~~~

.. autoapisummary::

   nebula.addons.trustworthiness.calculation.dirname
   nebula.addons.trustworthiness.calculation.logger
   nebula.addons.trustworthiness.calculation.R_L1
   nebula.addons.trustworthiness.calculation.R_L2
   nebula.addons.trustworthiness.calculation.R_LI


.. py:data:: dirname

   

.. py:data:: logger

   

.. py:data:: R_L1
   :value: 40

   

.. py:data:: R_L2
   :value: 2

   

.. py:data:: R_LI
   :value: 0.1

   

.. py:function:: get_mapped_score(score_key, score_map)

   Finds the score by the score_key in the score_map.

   :param score_key: The key to look up in the score_map.
   :type score_key: string
   :param score_map: The score map defined in the eval_metrics.json file.
   :type score_map: dict

   :returns: The normalized score of [0, 1].
   :rtype: float


.. py:function:: get_normalized_scores(scores)

   Calculates the normalized scores of a list.

   :param scores: The values that will be normalized.
   :type scores: list

   :returns: The normalized list.
   :rtype: list


.. py:function:: get_range_score(value, ranges, direction='asc')

   Maps the value to a range and gets the score by the range and direction.

   :param value: The input score.
   :type value: int
   :param ranges: The ranges defined.
   :type ranges: list
   :param direction: Asc means the higher the range the higher the score, desc means otherwise.
   :type direction: string

   :returns: The normalized score of [0, 1].
   :rtype: float


.. py:function:: get_map_value_score(score_key, score_map)

   Finds the score by the score_key in the score_map and returns the value.

   :param score_key: The key to look up in the score_map.
   :type score_key: string
   :param score_map: The score map defined in the eval_metrics.json file.
   :type score_map: dict

   :returns: The score obtained in the score_map.
   :rtype: float


.. py:function:: get_true_score(value, direction)

   Returns the negative of the value if direction is 'desc', otherwise returns value.

   :param value: The input score.
   :type value: int
   :param direction: Asc means the higher the range the higher the score, desc means otherwise.
   :type direction: string

   :returns: The score obtained.
   :rtype: float


.. py:function:: get_scaled_score(value, scale, direction)

   Maps a score of a specific scale into the scale between zero and one.

   :param value: The raw value of the metric.
   :type value: int or float
   :param scale: List containing the minimum and maximum value the value can fall in between.
   :type scale: list

   :returns: The normalized score of [0, 1].
   :rtype: float


.. py:function:: get_value(value)

   Get the value of a metric.

   :param value: The value of the metric.
   :type value: float

   :returns: The value of the metric.
   :rtype: float


.. py:function:: check_properties(*args)

   Check if all the arguments have values.

   :param args: All the arguments.
   :type args: list

   :returns: The mean of arguments that have values.
   :rtype: float


.. py:function:: get_cv(list=None, std=None, mean=None)

   Get the coefficient of variation.

   :param list: List in which the coefficient of variation will be calculated.
   :type list: list
   :param std: Standard deviation of a list.
   :type std: float
   :param mean: Mean of a list.
   :type mean: float

   :returns: The coefficient of variation calculated.
   :rtype: float


.. py:function:: get_global_privacy_risk(dp, epsilon, n)

   Calculates the global privacy risk by epsilon and the number of clients.

   :param dp: Indicates if differential privacy is used or not.
   :type dp: bool
   :param epsilon: The epsilon value.
   :type epsilon: int
   :param n: The number of clients in the scenario.
   :type n: int

   :returns: The global privacy risk.
   :rtype: float


.. py:function:: get_elapsed_time(scenario)

   Calculates the elapsed time during the execution of the scenario.

   :param scenario: Scenario required.
   :type scenario: object

   :returns: The elapsed time.
   :rtype: float


.. py:function:: get_bytes_models(models_files)

   Calculates the mean bytes of the final models of the nodes.

   :param models_files: List of final models.
   :type models_files: list

   :returns: The mean bytes of the models.
   :rtype: float


.. py:function:: get_bytes_sent_recv(bytes_sent_files, bytes_recv_files)

   Calculates the mean bytes sent and received of the nodes.

   :param bytes_sent_files: Files that contain the bytes sent of the nodes.
   :type bytes_sent_files: list
   :param bytes_recv_files: Files that contain the bytes received of the nodes.
   :type bytes_recv_files: list

   :returns: The total bytes sent, the total bytes received, the mean bytes sent and the mean bytes received of the nodes.
   :rtype: 4-tupla


.. py:function:: get_avg_loss_accuracy(loss_files, accuracy_files)

   Calculates the mean accuracy and loss models of the nodes.

   :param loss_files: Files that contain the loss of the models of the nodes.
   :type loss_files: list
   :param accuracy_files: Files that contain the acurracies of the models of the nodes.
   :type accuracy_files: list

   :returns: The mean loss of the models, the mean accuracies of the models, the standard deviation of the accuracies of the models.
   :rtype: 3-tupla


.. py:function:: get_feature_importance_cv(model, test_sample)

   Calculates the coefficient of variation of the feature importance.

   :param model: The model.
   :type model: object
   :param test_sample: One test sample to calculate the feature importance.
   :type test_sample: object

   :returns: The coefficient of variation of the feature importance.
   :rtype: float


.. py:function:: get_clever_score(model, test_sample, nb_classes, learning_rate)

   Calculates the CLEVER score.

   :param model: The model.
   :type model: object
   :param test_sample: One test sample to calculate the CLEVER score.
   :type test_sample: object
   :param nb_classes: The nb_classes of the model.
   :type nb_classes: int
   :param learning_rate: The learning rate of the model.
   :type learning_rate: float

   :returns: The CLEVER score.
   :rtype: float


.. py:function:: stop_emissions_tracking_and_save(tracker, outdir, emissions_file, role, workload, sample_size = 0)

   Stops emissions tracking object from CodeCarbon and saves relevant information to emissions.csv file.

   :param tracker: The emissions tracker object holding information.
   :type tracker: object
   :param outdir: The path of the output directory of the experiment.
   :type outdir: str
   :param emissions_file: The path to the emissions file.
   :type emissions_file: str
   :param role: Either client or server depending on the role.
   :type role: str
   :param workload: Either aggregation or training depending on the workload.
   :type workload: str
   :param sample_size: The number of samples used for training, if aggregation 0.
   :type sample_size: int



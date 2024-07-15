nebula.addons.trustworthiness.utils
===================================

.. py:module:: nebula.addons.trustworthiness.utils


Attributes
----------

.. autoapisummary::

   nebula.addons.trustworthiness.utils.hashids
   nebula.addons.trustworthiness.utils.logger
   nebula.addons.trustworthiness.utils.dirname


Functions
---------

.. autoapisummary::

   nebula.addons.trustworthiness.utils.count_class_samples
   nebula.addons.trustworthiness.utils.get_entropy
   nebula.addons.trustworthiness.utils.read_csv
   nebula.addons.trustworthiness.utils.check_field_filled
   nebula.addons.trustworthiness.utils.get_input_value
   nebula.addons.trustworthiness.utils.get_value_from_path
   nebula.addons.trustworthiness.utils.write_results_json
   nebula.addons.trustworthiness.utils.save_results_csv


Module Contents
---------------

.. py:data:: hashids

.. py:data:: logger

.. py:data:: dirname

.. py:function:: count_class_samples(scenario_name, dataloaders_files)

   Counts the number of samples by class.

   :param scenario_name: Name of the scenario.
   :type scenario_name: string
   :param dataloaders_files: Files that contain the dataloaders.
   :type dataloaders_files: list


.. py:function:: get_entropy(client_id, scenario_name, dataloader)

   Get the entropy of each client in the scenario.

   :param client_id: The client id.
   :type client_id: int
   :param scenario_name: Name of the scenario.
   :type scenario_name: string
   :param dataloaders_files: Files that contain the dataloaders.
   :type dataloaders_files: list


.. py:function:: read_csv(filename)

   Read a CSV file.

   :param filename: Name of the file.
   :type filename: string

   :returns: The CSV readed.
   :rtype: object


.. py:function:: check_field_filled(factsheet_dict, factsheet_path, value, empty='')

   Check if the field in the factsheet file is filled or not.

   :param factsheet_dict: The factshett dict.
   :type factsheet_dict: dict
   :param factsheet_path: The factsheet field to check.
   :type factsheet_path: list
   :param value: The value to add in the field.
   :type value: float
   :param empty: If the value could not be appended, the empty string is returned.
   :type empty: string

   :returns: The value added in the factsheet or empty if the value could not be appened
   :rtype: float


.. py:function:: get_input_value(input_docs, inputs, operation)

   Gets the input value from input document and apply the metric operation on the value.

   :param inputs_docs: The input document map.
   :type inputs_docs: map
   :param inputs: All the inputs.
   :type inputs: list
   :param operation: The metric operation.
   :type operation: string

   :returns: The metric value
   :rtype: float


.. py:function:: get_value_from_path(input_doc, path)

   Gets the input value from input document by path.

   :param inputs_doc: The input document map.
   :type inputs_doc: map
   :param path: The field name of the input value of interest.
   :type path: string

   :returns: The input value from the input document
   :rtype: float


.. py:function:: write_results_json(out_file, dict)

   Writes the result to JSON.

   :param out_file: The output file.
   :type out_file: string
   :param dict: The object to be witten into JSON.
   :type dict: dict

   :returns: The input value from the input document
   :rtype: float


.. py:function:: save_results_csv(scenario_name, id, bytes_sent, bytes_recv, accuracy, loss, finish)


nebula.config.mender
====================

.. py:module:: nebula.config.mender


Classes
-------

.. autoapisummary::

   nebula.config.mender.Mender


Module Contents
---------------

.. py:class:: Mender

   .. py:method:: get_token()


   .. py:method:: renew_token()


   .. py:method:: generate_artifact(type_artifact, artifact_name, device_type, file_path)
      :staticmethod:



   .. py:method:: get_artifacts()


   .. py:method:: upload_artifact(artifact_path, description)


   .. py:method:: deploy_artifact_device(artifact_name, device)


   .. py:method:: deploy_artifact_list(artifact_name, devices)


   .. py:method:: get_info_deployment(deployment_id)


   .. py:method:: get_my_info()


   .. py:method:: get_devices()


   .. py:method:: get_devices_by_group(group)


   .. py:method:: get_info_device(device_id)


   .. py:method:: get_connected_device(device_id)



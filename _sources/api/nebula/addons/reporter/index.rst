nebula.addons.reporter
======================

.. py:module:: nebula.addons.reporter


Classes
-------

.. autoapisummary::

   nebula.addons.reporter.Reporter


Module Contents
---------------

.. py:class:: Reporter(config, trainer, cm)

   .. py:attribute:: config


   .. py:attribute:: trainer


   .. py:attribute:: cm


   .. py:attribute:: frequency


   .. py:attribute:: grace_time


   .. py:attribute:: data_queue


   .. py:attribute:: url


   .. py:attribute:: counter
      :value: 0



   .. py:attribute:: first_net_metrics
      :value: True



   .. py:attribute:: prev_bytes_sent
      :value: 0



   .. py:attribute:: prev_bytes_recv
      :value: 0



   .. py:attribute:: prev_packets_sent
      :value: 0



   .. py:attribute:: prev_packets_recv
      :value: 0



   .. py:attribute:: acc_bytes_sent
      :value: 0



   .. py:attribute:: acc_bytes_recv
      :value: 0



   .. py:attribute:: acc_packets_sent
      :value: 0



   .. py:attribute:: acc_packets_recv
      :value: 0



   .. py:method:: enqueue_data(name, value)
      :async:



   .. py:method:: start()
      :async:



   .. py:method:: run_reporter()
      :async:



   .. py:method:: report_scenario_finished()
      :async:




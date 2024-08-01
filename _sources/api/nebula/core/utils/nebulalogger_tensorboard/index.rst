nebula.core.utils.nebulalogger_tensorboard
==========================================

.. py:module:: nebula.core.utils.nebulalogger_tensorboard


Classes
-------

.. autoapisummary::

   nebula.core.utils.nebulalogger_tensorboard.NebulaTensorBoardLogger


Module Contents
---------------

.. py:class:: NebulaTensorBoardLogger(scenario_start_time, *args, **kwargs)

   Bases: :py:obj:`lightning.pytorch.loggers.TensorBoardLogger`


   .. py:attribute:: scenario_start_time


   .. py:attribute:: local_step
      :value: 0



   .. py:attribute:: global_step
      :value: 0



   .. py:method:: get_step()


   .. py:method:: log_data(data, step=None)


   .. py:method:: log_metrics(metrics, step=None)


   .. py:method:: log_figure(figure, step=None, name=None)



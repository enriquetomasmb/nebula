:py:mod:`nebula.core.utils.nebulalogger_tensorboard`
====================================================

.. py:module:: nebula.core.utils.nebulalogger_tensorboard


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.utils.nebulalogger_tensorboard.NebulaTensorBoardLogger




.. py:class:: NebulaTensorBoardLogger(scenario_start_time, *args, **kwargs)


   Bases: :py:obj:`lightning.pytorch.loggers.TensorBoardLogger`

   .. py:method:: get_step()


   .. py:method:: log_data(data, step=None)


   .. py:method:: log_metrics(metrics, step=None)


   .. py:method:: log_figure(figure, step=None, name=None)




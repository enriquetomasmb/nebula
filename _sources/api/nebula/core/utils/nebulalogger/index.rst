nebula.core.utils.nebulalogger
==============================

.. py:module:: nebula.core.utils.nebulalogger


Classes
-------

.. autoapisummary::

   nebula.core.utils.nebulalogger.NebulaLogger


Module Contents
---------------

.. py:class:: NebulaLogger(config, engine, scenario_start_time, *args, **kwargs)

   Bases: :py:obj:`aim.pytorch_lightning.AimLogger`


   .. py:attribute:: config


   .. py:attribute:: engine


   .. py:attribute:: scenario_start_time


   .. py:attribute:: local_step
      :value: 0



   .. py:attribute:: global_step
      :value: 0



   .. py:method:: finalize(status = '')


   .. py:method:: get_step()


   .. py:method:: log_data(data, step=None)


   .. py:method:: log_figure(figure, step=None, name=None)



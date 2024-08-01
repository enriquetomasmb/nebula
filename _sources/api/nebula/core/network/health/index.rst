nebula.core.network.health
==========================

.. py:module:: nebula.core.network.health


Classes
-------

.. autoapisummary::

   nebula.core.network.health.Health


Module Contents
---------------

.. py:class:: Health(addr, config, cm)

   .. py:attribute:: addr


   .. py:attribute:: config


   .. py:attribute:: cm


   .. py:attribute:: period


   .. py:attribute:: alive_interval


   .. py:attribute:: check_alive_interval


   .. py:attribute:: timeout


   .. py:method:: start()
      :async:



   .. py:method:: run_send_alive()
      :async:



   .. py:method:: run_check_alive()
      :async:



   .. py:method:: alive(source)
      :async:




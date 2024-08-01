nebula.addons.mobility
======================

.. py:module:: nebula.addons.mobility


Classes
-------

.. autoapisummary::

   nebula.addons.mobility.Mobility


Module Contents
---------------

.. py:class:: Mobility(config, cm)

   .. py:attribute:: config


   .. py:attribute:: cm


   .. py:attribute:: grace_time


   .. py:attribute:: period


   .. py:attribute:: mobility


   .. py:attribute:: mobility_type


   .. py:attribute:: radius_federation


   .. py:attribute:: scheme_mobility


   .. py:attribute:: round_frequency


   .. py:attribute:: max_distance_with_direct_connections
      :value: 300



   .. py:attribute:: max_movement_random_strategy
      :value: 100



   .. py:attribute:: max_movement_nearest_strategy
      :value: 100



   .. py:attribute:: max_initiate_approximation


   .. py:attribute:: network_conditions


   .. py:attribute:: current_network_conditions


   .. py:attribute:: mobility_msg


   .. py:property:: round


   .. py:method:: start()
      :async:



   .. py:method:: run_mobility()
      :async:



   .. py:method:: change_geo_location_random_strategy(latitude, longitude)
      :async:



   .. py:method:: change_geo_location_nearest_neighbor_strategy(distance, latitude, longitude, neighbor_latitude, neighbor_longitude)
      :async:



   .. py:method:: set_geo_location(latitude, longitude)
      :async:



   .. py:method:: change_geo_location()
      :async:



   .. py:method:: change_connections_based_on_distance()
      :async:



   .. py:method:: change_connections()
      :async:




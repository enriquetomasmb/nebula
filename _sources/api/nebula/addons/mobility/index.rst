:py:mod:`nebula.addons.mobility`
================================

.. py:module:: nebula.addons.mobility


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.addons.mobility.Mobility




.. py:class:: Mobility(config, cm)


   Bases: :py:obj:`threading.Thread`

   A class that represents a thread of control.

   This class can be safely subclassed in a limited fashion. There are two ways
   to specify the activity: by passing a callable object to the constructor, or
   by overriding the run() method in a subclass.


   .. py:property:: round


   .. py:method:: run()

      Method representing the thread's activity.

      You may override this method in a subclass. The standard run() method
      invokes the callable object passed to the object's constructor as the
      target argument, if any, with sequential and keyword arguments taken
      from the args and kwargs arguments, respectively.



   .. py:method:: change_geo_location_random_strategy(latitude, longitude)


   .. py:method:: change_geo_location_nearest_neighbor_strategy(distance, latitude, longitude, neighbor_latitude, neighbor_longitude)


   .. py:method:: set_geo_location(latitude, longitude)


   .. py:method:: change_geo_location()


   .. py:method:: change_connections_based_on_distance()


   .. py:method:: change_connections()
      :async:




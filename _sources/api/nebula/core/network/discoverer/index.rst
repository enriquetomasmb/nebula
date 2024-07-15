nebula.core.network.discoverer
==============================

.. py:module:: nebula.core.network.discoverer


Classes
-------

.. autoapisummary::

   nebula.core.network.discoverer.Discoverer


Module Contents
---------------

.. py:class:: Discoverer(addr, config, cm)

   Bases: :py:obj:`threading.Thread`


   A class that represents a thread of control.

   This class can be safely subclassed in a limited fashion. There are two ways
   to specify the activity: by passing a callable object to the constructor, or
   by overriding the run() method in a subclass.



   .. py:method:: run()

      Method representing the thread's activity.

      You may override this method in a subclass. The standard run() method
      invokes the callable object passed to the object's constructor as the
      target argument, if any, with sequential and keyword arguments taken
      from the args and kwargs arguments, respectively.




   .. py:method:: run_discover()
      :async:




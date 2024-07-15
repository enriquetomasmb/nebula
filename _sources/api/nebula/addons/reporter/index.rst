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

   Bases: :py:obj:`threading.Thread`


   A class that represents a thread of control.

   This class can be safely subclassed in a limited fashion. There are two ways
   to specify the activity: by passing a callable object to the constructor, or
   by overriding the run() method in a subclass.



   .. py:method:: enqueue_data(name, value)


   .. py:method:: run()

      Method representing the thread's activity.

      You may override this method in a subclass. The standard run() method
      invokes the callable object passed to the object's constructor as the
      target argument, if any, with sequential and keyword arguments taken
      from the args and kwargs arguments, respectively.




   .. py:method:: report_scenario_finished()



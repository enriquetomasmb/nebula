nebula.core.eventmanager
========================

.. py:module:: nebula.core.eventmanager


Classes
-------

.. autoapisummary::

   nebula.core.eventmanager.EventManager


Functions
---------

.. autoapisummary::

   nebula.core.eventmanager.event_handler


Module Contents
---------------

.. py:function:: event_handler(message_type, action)

   Decorator for registering an event handler.


.. py:class:: EventManager(default_callbacks=None)

   .. py:method:: register_event(handler_info, callback)

      Records a callback for a specific event.



   .. py:method:: unregister_event(handler_info, callback)

      Unregisters a previously registered callback for an event.



   .. py:method:: trigger_event(source, message, *args, **kwargs)
      :async:


      Triggers an event, executing all associated callbacks.



   .. py:method:: get_event_callbacks(event_name)
      :async:


      Returns the callbacks for a specific event.



   .. py:method:: get_event_callbacks_names()

      Returns the names of the registered events.




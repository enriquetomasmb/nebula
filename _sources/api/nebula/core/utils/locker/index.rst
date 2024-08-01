nebula.core.utils.locker
========================

.. py:module:: nebula.core.utils.locker


Classes
-------

.. autoapisummary::

   nebula.core.utils.locker.Locker


Module Contents
---------------

.. py:class:: Locker(name, verbose=True, async_lock=False, *args, **kwargs)

   .. py:method:: acquire(*args, **kwargs)


   .. py:method:: release(*args, **kwargs)


   .. py:method:: locked()


   .. py:method:: acquire_async(*args, **kwargs)
      :async:



   .. py:method:: release_async(*args, **kwargs)
      :async:



   .. py:method:: locked_async()
      :async:




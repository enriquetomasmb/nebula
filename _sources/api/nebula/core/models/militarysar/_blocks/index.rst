:orphan:

:py:mod:`nebula.core.models.militarysar._blocks`
================================================

.. py:module:: nebula.core.models.militarysar._blocks


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.militarysar._blocks.BaseBlock
   nebula.core.models.militarysar._blocks.DenseBlock
   nebula.core.models.militarysar._blocks.Conv2DBlock




.. py:class:: BaseBlock


   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x)



.. py:class:: DenseBlock(shape, **params)


   Bases: :py:obj:`BaseBlock`


.. py:class:: Conv2DBlock(shape, stride, padding='same', **params)


   Bases: :py:obj:`BaseBlock`



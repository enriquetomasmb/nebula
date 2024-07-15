nebula.addons.attacks.poisoning.datapoison
==========================================

.. py:module:: nebula.addons.attacks.poisoning.datapoison


Functions
---------

.. autoapisummary::

   nebula.addons.attacks.poisoning.datapoison.datapoison
   nebula.addons.attacks.poisoning.datapoison.add_x_to_image
   nebula.addons.attacks.poisoning.datapoison.poison_to_nlp_rawdata


Module Contents
---------------

.. py:function:: datapoison(dataset, indices, poisoned_persent, poisoned_ratio, targeted=False, target_label=3, noise_type='salt')

   Function to add random noise of various types to the dataset.


.. py:function:: add_x_to_image(img)

   Add a 10*10 pixels X at the top-left of an image


.. py:function:: poison_to_nlp_rawdata(text_data, poisoned_ratio)

   for NLP data, change the word vector to 0 with p=poisoned_ratio



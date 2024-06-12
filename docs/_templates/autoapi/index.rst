API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. [#f1] Created by `Enrique Tomás Martínez Beltrán <https://www.enriquetomasmb.com/>`_

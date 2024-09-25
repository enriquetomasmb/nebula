nebula.controller
=================

.. py:module:: nebula.controller


Attributes
----------

.. autoapisummary::

   nebula.controller.log_console_format
   nebula.controller.console_handler


Classes
-------

.. autoapisummary::

   nebula.controller.TermEscapeCodeFormatter
   nebula.controller.NebulaEventHandler
   nebula.controller.Controller


Functions
---------

.. autoapisummary::

   nebula.controller.signal_handler


Module Contents
---------------

.. py:class:: TermEscapeCodeFormatter(fmt=None, datefmt=None, style='%', validate=True)

   Bases: :py:obj:`logging.Formatter`


   Formatter instances are used to convert a LogRecord to text.

   Formatters need to know how a LogRecord is constructed. They are
   responsible for converting a LogRecord to (usually) a string which can
   be interpreted by either a human or an external system. The base Formatter
   allows a formatting string to be specified. If none is supplied, the
   style-dependent default value, "%(message)s", "{message}", or
   "${message}", is used.

   The Formatter can be initialized with a format string which makes use of
   knowledge of the LogRecord attributes - e.g. the default value mentioned
   above makes use of the fact that the user's message and arguments are pre-
   formatted into a LogRecord's message attribute. Currently, the useful
   attributes in a LogRecord are described by:

   %(name)s            Name of the logger (logging channel)
   %(levelno)s         Numeric logging level for the message (DEBUG, INFO,
                       WARNING, ERROR, CRITICAL)
   %(levelname)s       Text logging level for the message ("DEBUG", "INFO",
                       "WARNING", "ERROR", "CRITICAL")
   %(pathname)s        Full pathname of the source file where the logging
                       call was issued (if available)
   %(filename)s        Filename portion of pathname
   %(module)s          Module (name portion of filename)
   %(lineno)d          Source line number where the logging call was issued
                       (if available)
   %(funcName)s        Function name
   %(created)f         Time when the LogRecord was created (time.time()
                       return value)
   %(asctime)s         Textual time when the LogRecord was created
   %(msecs)d           Millisecond portion of the creation time
   %(relativeCreated)d Time in milliseconds when the LogRecord was created,
                       relative to the time the logging module was loaded
                       (typically at application startup time)
   %(thread)d          Thread ID (if available)
   %(threadName)s      Thread name (if available)
   %(process)d         Process ID (if available)
   %(message)s         The result of record.getMessage(), computed just as
                       the record is emitted


   .. py:method:: format(record)

      Format the specified record as text.

      The record's attribute dictionary is used as the operand to a
      string formatting operation which yields the returned string.
      Before formatting the dictionary, a couple of preparatory steps
      are carried out. The message attribute of the record is computed
      using LogRecord.getMessage(). If the formatting string uses the
      time (as determined by a call to usesTime(), formatTime() is
      called to format the event time. If there is exception information,
      it is formatted using formatException() and appended to the message.



.. py:data:: log_console_format
   :value: '[%(levelname)s] - %(asctime)s - Controller - %(message)s'


.. py:data:: console_handler

.. py:function:: signal_handler(sig, frame)

.. py:class:: NebulaEventHandler

   Bases: :py:obj:`watchdog.events.PatternMatchingEventHandler`


   NebulaEventHandler handles file system events for .sh scripts.

   This class monitors the creation, modification, and deletion of .sh scripts
   in a specified directory.


   .. py:attribute:: patterns
      :value: ['*.sh', '*.ps1']



   .. py:attribute:: last_processed


   .. py:attribute:: timeout_ns


   .. py:attribute:: processing_files


   .. py:attribute:: lock


   .. py:method:: on_created(event)

      Handles the event when a file is created.



   .. py:method:: on_deleted(event)

      Handles the event when a file is deleted.



   .. py:method:: run_script(script)


   .. py:method:: kill_script_processes(pids_file)


.. py:class:: Controller(args)

   .. py:attribute:: scenario_name


   .. py:attribute:: start_date_scenario
      :value: None



   .. py:attribute:: federation


   .. py:attribute:: topology


   .. py:attribute:: waf_port


   .. py:attribute:: frontend_port


   .. py:attribute:: grafana_port


   .. py:attribute:: loki_port


   .. py:attribute:: statistics_port


   .. py:attribute:: simulation


   .. py:attribute:: config_dir


   .. py:attribute:: test


   .. py:attribute:: log_dir


   .. py:attribute:: cert_dir


   .. py:attribute:: env_path


   .. py:attribute:: production


   .. py:attribute:: advanced_analytics


   .. py:attribute:: matrix


   .. py:attribute:: root_path


   .. py:attribute:: host_platform


   .. py:attribute:: network_subnet


   .. py:attribute:: network_gateway


   .. py:attribute:: config


   .. py:attribute:: topologymanager
      :value: None



   .. py:attribute:: n_nodes
      :value: 0



   .. py:attribute:: mender


   .. py:attribute:: use_blockchain


   .. py:attribute:: gpu_available
      :value: False



   .. py:method:: start()


   .. py:method:: run_waf()


   .. py:method:: run_frontend()


   .. py:method:: run_test()


   .. py:method:: stop_frontend()
      :staticmethod:



   .. py:method:: stop_network()
      :staticmethod:



   .. py:method:: stop_waf()
      :staticmethod:



   .. py:method:: stop()
      :staticmethod:




# logger.py
"""
LOGGER SETUP FOR VI_RAG (DICTCONFIG VERSION)

----------------------------------------
WHAT THIS FILE DOES
----------------------------------------
This file defines how logging is activated for the project.

We load configuration from settings.py and apply it
WHEN explicitly requested.

----------------------------------------
HOW TO USE
----------------------------------------
Call the setup function ONCE at application start:

    from logger import setup_logging
    setup_logging()

After that, all modules can safely use logging.

----------------------------------------
SIMPLE IDEA
----------------------------------------
settings.py → defines logging
logger.py   → ACTIVATES logging (on demand)
your files  → use logging

----------------------------------------
WHY USE A FUNCTION?
----------------------------------------

✔ Avoids side effects on import
✔ Gives explicit control over when logging starts
✔ Prevents accidental reconfiguration
✔ Matches best practice in larger applications

----------------------------------------
IMPORTANT
----------------------------------------
✔ Do NOT create loggers here
✔ Do NOT define handlers here
✔ Only apply configuration

----------------------------------------
ADVANCED BEHAVIOUR (IMPORTANT CONCEPT)
----------------------------------------

✔ If you DO NOT explicitly name a logger:

    logging.getLogger(__name__)

Python automatically creates a logger based on the
module path (e.g. SeqKitSTP.core.utils)

✔ It is best practice to initialise logging at the
root of your installable package (e.g. SeqKitSTP)
so all modules inherit consistent behaviour

✔ Hierarchy behaviour:
- Loggers inherit configuration from their parents
- Ultimately, everything inherits from the ROOT logger
  unless explicitly overridden

✔ What this means in practice:

- If you have NOT explicitly defined a logger:
  → it inherits level, handlers, and formatting from ROOT

- So by default:
  → all unnamed loggers follow top-level configuration

✔ Best practice:
- Use __name__ in modules:
    logger = logging.getLogger(__name__)

✔ Granular control:
- In larger systems (e.g. web apps):
  you can override behaviour per module/package
  using LOGGING_CONFIG["loggers"]

----------------------------------------
TEACHING TAKEAWAY
----------------------------------------

✔ Named logger in config → explicit control
✔ No named logger        → inherits from root
✔ setup_logging()        → single point of activation

→ Explicit, predictable, scalable logging
"""

import logging.config
from log_settings import LOGGING_CONFIG


# --------------------------------------------------
# LOGGING ACTIVATION FUNCTION
# --------------------------------------------------
def setup_logging():
    """
    Apply the logging configuration.

    This should be called ONCE at application startup.

    After this:
    ✔ Logging is globally configured
    ✔ All modules can use logging.getLogger()
    ✔ Hierarchical inheritance works automatically
    """

    logging.config.dictConfig(LOGGING_CONFIG)
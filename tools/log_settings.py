# settings.py
"""
LOGGING CONFIGURATION
"""

import os

# --------------------------------------------------
# STEP 1: Define default log file (hidden, home directory)
# --------------------------------------------------
curr_dir = os.path.abspath("__file__")
LOG_FILE = os.path.join(curr_dir, "logs", "importation.log")

# --------------------------------------------------
# STEP 3: Ensure directory exists
# --------------------------------------------------
log_dir = os.path.join(curr_dir, "logs")
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

# --------------------------------------------------
# STEP 4: Logging configuration
# --------------------------------------------------
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    # --------------------------------------------------
    # FORMATTERS
    # --------------------------------------------------
    "formatters": {
        "standard": {
            "format": (
                "%(asctime)s | %(levelname)s | %(name)s | "
                "%(filename)s:%(lineno)d | %(message)s"
            )
        }
    },

    # --------------------------------------------------
    # HANDLERS
    # --------------------------------------------------
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },

        # --------------------------------------------------
        # ROTATING FILE HANDLER (UPDATED)
        # --------------------------------------------------
        #
        # This replaces the basic FileHandler with a
        # RotatingFileHandler to prevent unlimited growth.
        #
        "file": {
            "class": "logging.handlers.RotatingFileHandler",

            # Only store ERROR and CRITICAL messages in file
            # (keeps disk usage focused on important events)
            "level": "DEBUG",

            "formatter": "standard",
            "filename": LOG_FILE,

            # --------------------------------------------------
            # ROTATION SETTINGS (IMPORTANT TEACHING POINT)
            # --------------------------------------------------
            #
            # maxBytes:
            # Maximum size of the log file BEFORE rotation happens.
            #
            # Example:
            # 1_048_576 bytes = 1 MB
            #
            # Here we set a small size for demonstration.
            #
            "maxBytes": 1024 * 100,  # 50 KB

            #
            # backupCount:
            # Number of OLD log files to retain.
            #
            # Example with backupCount = 3:
            #
            # Files will look like:
            #   .seqkitstp.log      (current active log)
            #   .seqkitstp.log.1    (most recent previous)
            #   .seqkitstp.log.2
            #   .seqkitstp.log.3    (oldest retained)
            #
            # When rotation occurs:
            #   .log.3 is deleted
            #   .log.2 → becomes .log.3
            #   .log.1 → becomes .log.2
            #   .log   → becomes .log.1
            #   new empty .log is created
            #
            "backupCount": 5,

            # Optional: ensures file opens safely even if reused
            "encoding": "utf-8",
        },
    },

    # ---------------------------------------------------------------------------------------
    # OPTIONAL: NAMED LOGGER (TEACHING EXAMPLE) - But easy to import into other tools by name
    # ---------------------------------------------------------------------------------------
    "loggers": {
        "VI_RAG": {

            # Capture everything at this level;
            # handlers will decide what gets written
            "level": "DEBUG",

            # Send logs to both console and rotating file
            "handlers": ["console", "file"],

            # Prevent duplication from parent/root loggers
            "propagate": False
        },
    },

    # --------------------------------------------------
    # ROOT LOGGER
    # --------------------------------------------------
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    }
}
"""VideoMetricsLab application package."""
import logging

# Silent until the app starts its log file (vmaf_app.core.app_log): library
# use and tests must not print the package's log lines to stderr.
logging.getLogger(__name__).addHandler(logging.NullHandler())

APP_NAME = "VideoMetricsLab"
__version__ = "1.2.1"

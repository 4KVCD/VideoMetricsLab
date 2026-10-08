"""VideoMetricsLab application package."""
import logging
import os

# numpy's OpenBLAS commits a 32 MB buffer for each thread it would use, one a
# CPU thread, when numpy is imported: 750 MB more private memory on a 24-thread
# CPU, in the app and again in each process it starts (vmaf_app.core.isolated),
# and 23 idle threads in each. The app makes no BLAS call (no matrix products
# or linear algebra), so one thread costs nothing. Set before numpy is
# imported, which every entry point does after this; processes the app starts
# inherit it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Silent until the app starts its log file (vmaf_app.core.app_log): library
# use and tests must not print the package's log lines to stderr.
logging.getLogger(__name__).addHandler(logging.NullHandler())

APP_NAME = "VideoMetricsLab"
__version__ = "2.0"

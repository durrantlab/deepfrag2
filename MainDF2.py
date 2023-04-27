"""Runs DeepFrag2."""

import time
import logging
from apps.deepfrag.run import function_to_run_deepfrag

if __name__ == "__main__":
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    print("Hello DeepFrag")
    start_time = time.time()
    function_to_run_deepfrag()
    final_time = time.time()
    print(
        f"Successful DeepFrag execution in: {str(final_time - start_time)} seconds"
    )

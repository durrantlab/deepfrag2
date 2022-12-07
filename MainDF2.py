import time
from apps.deepfrag.run import function_2run_deepfrag

if __name__ == "__main__":
    print("Hello DeepFrag")
    start_time = time.time()
    function_2run_deepfrag()
    final_time = time.time()
    print("Successful DeepFrag execution in: " + str(final_time - start_time) + " seconds")

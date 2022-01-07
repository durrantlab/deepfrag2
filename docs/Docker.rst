Docker
======

Overview
--------

Collagen comes with Docker support. You can use the `docker/manager.py` tool to build and run Docker images:

List available images
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ ./manager list

    cpu: CPU only
    cu111: CUDA 11.1

Build an image
^^^^^^^^^^^^^^

.. code-block:: bash

    $ ./manager build cpu

    ...
    => [10/11] COPY collagen /home/collagen                                                                                                            0.0s
    => [11/11] WORKDIR /home/collagen                                                                                                                  0.0s
    => exporting to image                                                                                                                              0.0s
    => => exporting layers                                                                                                                             0.0s
    => => writing image sha256:10abb2aec3e09c1c2e71454461ca81d31f32ef1b8deaad62ab71d32a608df0cd                                                        0.0s
    => => naming to docker.io/library/collagen_cpu

Run an image
^^^^^^^^^^^^

By default, `manager` will mount the apps folder (`collagen/apps`) at `/mnt/apps`. You can override the default apps path with the `--apps` parameter.

.. code-block:: bash

    $ ./manager run cpu

    root@85748b57b7fa:/mnt/apps#

You can also specify a `data` and `checkpoints` folder (mounted at `/mnt/data` and `/mnt/checkpoints` respectively):

.. code-block:: bash

    $ ./manager run --data ~/my_data --checkpoints ~/my_checkpoints cpu

    root@85748b57b7fa:/mnt/apps# ls /mnt
    apps data checkpoints

By default, the docker container will shut down when you exit the interactive shell. To launch a long-running container, you can pass the `daemon` argument to the internal docker command with the `--extra` flag:

.. code-block:: bash

    $ ./manager run --extra="-d" cpu
    84486bc1faad8d10d732702cdff00b028245314793cbe358c30f33a226b89a80

    $ docker ps
    CONTAINER ID   IMAGE          COMMAND       CREATED              STATUS          PORTS     NAMES
    84486bc1faad   collagen_cpu   "/bin/bash"   About a minute ago   Up 59 seconds             goofy_edison

When running GPU-enabled containers, be sure to make GPUs visible to docker with the `--gpus` flag:

.. code-block:: bash

    $ ./manager run --extra="--gpus=1" cu111

    root@9ea4b8af439e:/mnt/apps# nvidia-smi -L
    GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-1a5ce867-f460-e562-a817-91c0cac3efc6)

Run unit tests
^^^^^^^^^^^^^^

.. code-block:: bash

    $ ./manager test cpu

    ...
    ----------------------------------------------------------------------
    Ran 1 test in 0.001s

    OK

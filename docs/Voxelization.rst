Voxelization
============

<name> provides GPU-accelerated voxelization utilities for converting molecular structures to 3D density tensors.

These tensors can be used in 3D convolutional architectures.

Voxel Parameters
----------------

Voxelation parameters are controlled by the :class:`atlas.data.voxelizer.VoxelParams` class. For example, we can instantiate a simple atomic-number-based voxelizer with the following parameters:

.. code-block:: python

    vp = VoxelParams(
        resolution=0.75,
        width=24,
        atom_featurizer=AtomicNumFeaturizer([1,6,7,8,16])
    )

.. autoclass:: atlas.data.voxelizer.VoxelParams

Single Voxelization
-------------------

The :class:`atlas.data.mol.Mol` class provides a ``voxelize`` method to generate a tensor for a single molecule:

.. automethod:: atlas.data.mol.Mol.voxelize

Batch Voxelization
------------------

It is often useful to construct a single tensor with a batch of voxelized molecules. The :class:`atlas.data.mol.Mol` class also provides a ``voxelize_into`` method which can facilitate in-place voxelization for an existing PyTorch tensor.

Note that if you invoke ``voxelize_into`` with ``cpu=False``, your tensor must be on the GPU (i.e. initialize with ``device='cuda'``). You can use :py:meth:`atlas.data.voxelizer.VoxelParams.tensor_size` to compute the target tensor size for a multi-batch tensor.

.. automethod:: atlas.data.mol.Mol.voxelize_into

PyTorch Voxelize Transform
--------------------------

For dealing with on-the-fly voxelation, you can also use the :class:`atlas.data.transforms.Voxelize` class as a Dataset transform. For example:

.. code-block:: python

    vp = VoxelParams(
        resolution=0.75,
        width=24,
        atom_featurizer=AtomicNumFeaturizer([1,6,7,8,15])
    )

    zinc = ZINCDatasetH5(
        './path/to/zinc.h5', 
        make_3D=True, 
        transform=transforms.Voxelize(vp, cpu=True)
    )

The Dataset will invoke :py:meth:`atlas.data.mol.Mol.voxelize` automatically upon loading the sample and this can be used with DataLoader to instantiate a tensor batch automatically, e.g.:

.. code-block:: python

    data = DataLoader(zinc, batch_size=16)

    print(next(iter(data)).shape)
    # torch.Size([16, 5, 24, 24, 24])

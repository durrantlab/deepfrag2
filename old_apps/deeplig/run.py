import sys
sys.path.append('../..')

from collagen.examples.voxel_to_fp_utils.train_utils import add_args_voxel_to_fp_model
from collagen.examples.deeplig import train

args = add_args_voxel_to_fp_model()

train.run(args)


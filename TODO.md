Re. learning rate, this is great:
https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method

**** When doing inference, get best validation checkpoint (not last).

I think a lot of my old code is not cruft. Good to do audit.

Is split_seed deterministic? Not specifying, should be set to "1" (default), but
seeing different counts in progress bar.

Pickle PDBs? What would be fastest?

"Training will start from the beginning of the next epoch. This can cause
unreliable results if further training is done, consider using an end of epoch
checkpoint."

Look into /dev/shm ramfs (to speed file load). Disk interface, but all in
memory. Virtual file system. Each epoch loads file again. Also, each fragment
loads multiple times. Can do it with docker, but could be tricky (must use host
/dev/shim). Could give speed up, but not sure.

Need to also project average vector (across rotations) into pca space. Not same
as aveaging pcas. And good to check that calculating correctly.

For closest (from average) could also output fingerprints.

Print out PCA variance accounted for by each fragment.


# Training goal

And what is the cardinal rule of neural network training? Whenever possible, use
a larger batch size.

Find where validation set is a nadir. If going back up again, indicates
overfitting. 

Learning rates between 1e-2 and 1e-4 are pretty typical. But remember that Adam
optimizer determins LR for each parameter, so perhaps limited utility.

Re. fine tuning, best to just see if something works at all using standard
parameters, and then to adjust hyperparameters to improve. If not getting any
improvement, hyperparameter tuning unlikely to make learning possible.

# Tips for determining if something is "learnable"

Try overfitting on a dataset of only a few examples. Loss goes to 0, val starts
to go up again.

# DONE

Ask Harrison to look over inference code.

Check out /mnt/Data/jdurrant/deepfrag2/apps/deepfrag/err.txt . These are ones you couldn't calculate properties for. Why?

Create moad cache using multiple processors?

Store splits, split seed in json file in working dir?

torch.backends.cudnn.benchmark = True

Due to changes in the batch size, you will have to tune your learning rate. You
can also incorporate learning rate warmup and learning rate decay. Apart from
that, you can use weight decay in your model.

https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html

If cache not given, check every.csv.cache.json

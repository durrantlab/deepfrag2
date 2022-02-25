Re. learning rate, this is great:
https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method

When doing inference, get best validation checkpoint (not last).

I think a lot of my old code is not cruft. Good to do audit.

Is split_seed deterministic? Not specifying, should be set to "1" (default), but
seeing different counts in progress bar.

"Training will start from the beginning of the next epoch. This can cause
unreliable results if further training is done, consider using an end of epoch
checkpoint."

Look into /dev/shm ramfs (to speed file load). Disk interface, but all in
memory. Virtual file system. Each epoch loads file again. Also, each fragment
loads multiple times. Can do it with docker, but could be tricky (must use host
/dev/shim). Could give speed up, but not sure.

Figure out how to not voxelize receptor when you just need the fragment smiles
string. Notes in the code.

Still need to set up inferance on user-provided PDB.

FORMAT OUTPUT BETTER: No [27*]. Make charges neural. No chirality.

What to aim for (orig model): https://mail.google.com/mail/u/0/#inbox/FMfcgzGmtrSFVRWFdpWQVbFnvJSdQGKj

When you use a saved split, is the order still randomized? That's important even
if its saved.

What about weighting loss by prevelance of fragment (so OH not overrepresented)?

When finding most simiar, you could find most similar to any of rotation
outputs. Maybe weighted by appearing in multiple lists?

# EASY IDEAS

What about only accepting fragment if it's among those in FDA-approved drugs (or clinical trials)?

SOMETHING TO TRY: Receptor + decorating fragments => Murkoscafold

What about filtering out fragments with more than a certain number of rotatable
bonds?

I think you could easily use this to come up with a loist of bioististers.

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

Also, connection point seems to be repeated een though it never shoul dbe.
(STILL A PROBLEM)

Clustering multiple-rotation outputs.

Need to redo high-res and high-affinity with new system.

Note: confirmed optimization (ignore dist atoms, etc.) doesn't impact accuracy
(top-k within 1 or 2 / 10000 when rot fixed = likely rounding error)

t-SNE. Method for dimensionality reduction that might work better than PCA.
Harrison brought up idea of training on that representation, instead of
RDKfingerprint. The thinking is that it might be "smoother." Something to
consider in terms of other fingerprint representations. NOTE: Looked into it, I
think t-SNE might be better for visualizing data too. Preserves local
neighborness, not global data structure like PCA.
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

Could it be that it is randomized, and that's why signal degrades on multiple
rotations?

Even when you fix the rotations, after a while predictionsPerRotation gives different vectors. Should always be giing the same vecotr (like it does at the start). There's something up there.

When pickling, could save only atoms within distance of cutoff (to reduce GPU
calcs and speed things further).

Also, no branching in numba re. accumulation type.

Pickle PDBs? What would be fastest?

Curently model has TOP1 accuracy of 21%, not 58%. Why? Run tests to see if you
gete similar value rotating only oonce, using fewer data examples. Need to be
able to test things quickly. (Fixed bug found in evaluation code.)

Could this error have something to do with the altered top-k accuracy? "UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 77. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`."

You need to get to the bottom of this:

/miniconda/lib/python3.8/site-packages/pytorch_lightning/utilities/data.py:56: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 16. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.

Also, does connection point (*) matter? NOTE: IT DOES (which is a good thing).

You might need to protonate the fragments before calculating rdkfingerprints.
For example, does [27*][O-] and [3*]O give the same fingerprint. NOTE: I have
confirmed that same FP regardless of proton. Also, [27*] vs [3*] doesn't matter
(number). Note that chirality also doesn't matter.

Apply May model to high-affinity and high-resolutions sets.

Go over top-k metric to understand. I think I now understand. Do a spot check
where you calculate it from the JSON file to confirm top-K calc is right.

Need to also project average vector (across rotations) into pca space. Not same
as aveaging pcas. And good to check that calculating correctly.

For closest (from average) could also output fingerprints.

Print out PCA variance accounted for by each fragment.

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

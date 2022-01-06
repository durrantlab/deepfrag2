Re. learning rate, this is great:
https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method

Ask Harrison to look over inference code.

When doing inference, get best validation checkpoint (not last).

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

Create moad cache using multiple processors?

Store splits, split seed in json file in working dir?

torch.backends.cudnn.benchmark = True

Due to changes in the batch size, you will have to tune your learning rate. You
can also incorporate learning rate warmup and learning rate decay. Apart from
that, you can use weight decay in your model.

https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html


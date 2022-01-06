# Collagen Apps

These apps all use the collagen framework. `deepfrag` is one the apps. Here's
how to run deepfrag:

```bash
python /PATH/TO/deepfrag2/apps/run_app.py deepfrag ./output_dir
```

You can also pass deepfrag parameters using a custom `params.json` file. It
should look something like this:

```json
{
     "csv": "every_smaller.csv",
     "wandb_project": "3aee432b3e7c672a3b2d2accf15b6b56a2770584"
}
```

`"csv"` indicates the location of the BindingMOAD csv file you wish to use. If
unspecified, deepfrag uses the `every.csv` file that comes with BindingMOAD
(i.e., it uses all entries).

`"wandb_project"` is an optional parameter that specifies the project id for use
with weights and biases. Good for logging training (highly recommended!).

To include these optional parameters, run the app like this:

```bash
python /PATH/TO/deepfrag2/apps/run_app.py deepfrag ./output_dir -p params.json
```

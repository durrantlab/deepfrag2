# Collagen apps

## Introduction

This directory allows users to run different collagen-powered apps via Docker.
Collagen is a framework for rapid prototyping machine-learning approaches to
computer-aided drug discovery.

We have nearly finished implementing `deepfrag` as a Collagen app. Here's how to
run deepfrag:

```bash
python /PATH/TO/deepfrag2/apps/run_app.py deepfrag ./output_dir
```

## User-defined parameters

You can also pass parameters to the dockerized app using a custom `params.json`
file. It should look something like this:

```json
{
     "csv": "every_smaller.csv",
     "wandb_project": "3aee432b3e7c672a3b2d2accf15b6b56a2770584",
     "batch_size": 512,
     "log_every_n_steps": 1
}
```

<!-- Helpful: https://www.tablesgenerator.com/markdown_tables -->

| Parameter             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `"csv"`               | Path to BindingMOAD csv. If unspecified, uses the `every.csv` (all entries). |
| `"wandb_project"`     | Project id for use with weights and biases.                                  |
| `"batch_size"`        | Batch size to use.                                                           |
| `"log_every_n_steps"` | How often to log metrics.                                                    |


This list of parameters may not be exhaustive, and acceptable parameters may
vary depending on the app. To see the default parameter values for each app,
look at `/PATH/TO/deepfrag2/apps/APP_NAME/defaults.json`.

To use these optional parameters (rather than just using the defaults), run the
app like this:

```bash
python /PATH/TO/deepfrag2/apps/run_app.py deepfrag ./output_dir -p params.json
```

## Directories

| Directory                           | Description                                                      |
|-------------------------------------|------------------------------------------------------------------|
| `./APP_NAME/` (e.g., `./deepfrag/`) | Information required to run app, including `defaults.json` file. |
| `./utils/`                          | Files required for running all apps.                             |
| `./utils/docker/`                   | Docker files.                                                    |

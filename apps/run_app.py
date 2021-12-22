import sys
import os
import argparse
import json
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("app_name", type=str, help="The app name.")
    parser.add_argument(
        "working_dir", type=str, help="The working directory, for checkpoints, etc."
    )
    parser.add_argument(
        "-p",
        "--params_json",
        type=str,
        help="A json file containing the app parameters. If omitted, uses default values.",
        default=None,
    )

    args = parser.parse_args()
    args.app_name = os.path.realpath(args.app_name)
    args.working_dir = os.path.realpath(args.working_dir)
    return args


def validate(args):
    # Do some validation
    if not os.path.exists(args.app_name):
        print("No app found at " + args.app_name)
        sys.exit(0)

    if not os.path.exists(args.app_name + os.sep + "run.py"):
        print("Required file missing: " + args.app_name + os.sep + "run.py")
        sys.exit(0)

    if not os.path.exists(args.app_name + os.sep + "defaults.json"):
        print("Required file missing: " + args.app_name + os.sep + "defaults.json")
        sys.exit(0)


def make_cur_app_dir(args):
    # Copy selected app to common name.
    os.system("rm -rf .cur_app; cp -r " + os.path.basename(args.app_name) + " .cur_app")

    # Construct commandline
    params = json.load(open("./.cur_app/defaults.json"))
    if args.params_json is not None:
        custom_params = json.load(open(args.params_json))
        for key in custom_params:
            params[key] = custom_params[key]
    params["default_root_dir"] = "/working/"

    exec = """echo START: $(date)
    python run.py """ + " ".join(
        ["--" + key + " " + str(params[key]) for key in params]
    )

    with open("./.cur_app/run.sh", "w") as f:
        f.write(exec)

    os.system("chmod +x ./.cur_app/run.sh")

    return params


args = get_args()
validate(args)
params = make_cur_app_dir(args)

# Save parameters to working directory.
date_str = datetime.now().strftime("%b-%d-%Y.%H-%M-%S")
with open(args.working_dir + os.sep + "params." + date_str + ".json", "w") as f:
    json.dump(params, f, indent=4)

# Build the docker image (every time).
os.system("cd ./utils/docker && docker build -t jdurrant/deeplig . ")

# Run the docker image
os.system(
    """cd ./utils/docker && \
docker run \
    --gpus all \
    -it --rm \
    --shm-size="2g" \
    --ipc=host \
    -v $(realpath ../../../):/mnt \
    -v $(realpath """
    + args.working_dir
    + """):/working \
    jdurrant/deeplig"""
)

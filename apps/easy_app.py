import sys
import os
import argparse
import json
from datetime import datetime
import time

print("\nThis script serves to simplify use on Durrant-lab computer systems. It may not work on other systems.\n")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CUR_APP_DIR = ".cur_app_" + str(time.time()).replace(".", "")

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
        help="A json file containing the app parameters. If omitted, uses app's default values.",
        default=None,
    )

    # Redundant (update elsewhere too)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Can be train or test. If train, trains the model. If test, runs inference on the test set. Defaults to train.",
        default="train",
    )

    args = parser.parse_args()
    args.name = args.app_name.replace("/", "")
    args.app_name = SCRIPT_DIR + "/" + args.app_name
    args.working_dir = os.path.realpath(args.working_dir)

    return args


def validate(args):
    # Do some validation
    if not os.path.exists(args.app_name):
        print("No app found at " + args.app_name)
        sys.exit(0)

    if not os.path.exists(args.app_name + "/run.py"):
        print("Required file missing: " + args.app_name + "/run.py")
        sys.exit(0)

    # TODO: JDD: Restore
    # if not os.path.exists(args.app_name + "/defaults.json"):
    #     print("Required file missing: " + args.app_name + "/defaults.json")
    #     sys.exit(0)

    if not os.path.exists(args.working_dir):
        os.system("mkdir " + args.working_dir)


def compile_parameters(args):
    # Get defaults
    params = json.load(open(SCRIPT_DIR + "/" + args.name + "/defaults.json"))

    # Merge in user specified
    if args.params_json is not None:
        custom_params = json.load(open(args.params_json))
        for key in custom_params:
            params[key] = custom_params[key]

    # If inference, make note of that too.
    params["mode"] = args.mode

    # Hard code some parameters
    params["default_root_dir"] = "/mnt/extra/checkpoints/"
    # params["cache"] = params["csv"] + ".cache.json"

    # Change csv to working dir if exists relative to this script.
    # TODO: JDD: Restore
    # if os.path.exists(params["csv"]):
    #     bsnm = os.path.basename(params["csv"])
    #     new_csv = args.working_dir + "/" + bsnm
    #     os.system("cp " + params["csv"] + " " + new_csv)
    #     params["csv"] = "/mnt/extra/" + bsnm

    # Change cache to working dir if exists relative to this script.
    # TODO: JDD: Restore
    # if os.path.exists(params["cache"]):
    #     bsnm = os.path.basename(params["cache"])
    #     new_cache = args.working_dir + "/" + bsnm
    #     os.system("cp " + params["cache"] + " " + new_cache)
    #     params["cache"] = "/mnt/extra/" + bsnm
    # else:
    #     # cache file doesn't exist. Update to be same as new csv file.
    #     # params["cache"] = params["csv"] + "." + args.name + ".cache.json"
    #     params["cache"] = params["csv"] + ".cache.json"

    return params

# TODO: JDD: Restore
# def make_cur_app_dir(args):
#     # Copy selected app to common name.
#     os.system(
#         "cd "
#         + SCRIPT_DIR
#         + "; rm -rf " + CUR_APP_DIR + "; cp -r "
#         + os.path.basename(args.app_name)
#         + " " + CUR_APP_DIR
#     )

#     # Construct commandline
#     params = compile_parameters(args)
#     exec = """echo START: $(date)
#     python run.py """ + " ".join(
#         ["--" + key + " " + str(params[key]) for key in params]
#     )

#     with open(SCRIPT_DIR + "/" + CUR_APP_DIR + "/run.sh", "w") as f:
#         f.write(exec)

#     os.system("cd " + SCRIPT_DIR + ";chmod +x ./" + CUR_APP_DIR + "/run.sh")

#     return params


args = get_args()
validate(args)

# TODO: JDD: Restore
# params = make_cur_app_dir(args)
params = compile_parameters(args)

# Save parameters to working directory.
# date_str = datetime.now().strftime("%b-%d-%Y.%H-%M-%S")
# with open(args.working_dir + "/params." + date_str + ".json", "w") as f:
with open(args.working_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# Save record of the .cur_app_* dirname being used
with open(args.working_dir + "/app_name.txt", "w") as f:
    f.write(args.name)

# Write the file to run in docker container.
with open(args.working_dir + "/run.sh", "w") as f:
    f.write("cd deepfrag\n")
    parts = [
        "--gpus 1",
        "--json_params /mnt/extra/params.json",
        "--csv " + params["csv"],
        "--data " + params["data"],
        "--max_voxels_in_memory " + str(params["max_voxels_in_memory"]),
    ]
    f.write("python run.py " + " ".join(parts))

def run(cmd):
    print("\n" + cmd + "\n")
    os.system(cmd)

# Build the docker image (every time).
run("cd ../ && ./manager build cu111")

# Run the docker image
prts = [
    '--data ' + os.path.abspath("../data/"),
    '--extra_dir ' + args.working_dir,
    '--cmd "/bin/bash /mnt/extra/run.sh "',
    # '--cmd "/bin/bash"',
    '--extra="--gpus=1"',
]
cmds = [
    "cd ../",
    './manager run ' + " ".join(prts) + ' cu111'
]
run("&&".join(cmds))


# os.system(
#     """cd """
#     + SCRIPT_DIR
#     + """/utils/docker && \
# docker run --gpus all -it --rm --shm-size="2g" --ipc=host \
#     -v $(realpath ../../../):/mnt \
#     -v $(realpath """
#     + args.working_dir
#     + """):/working \
#     jdurrant/deeplig"""
# )

# # Clean up
# os.system("rm -rf " + CUR_APP_DIR)

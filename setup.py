from setuptools import setup
import os

# It's important to make sure the pip and conda installations are consistent.
# This is admittedly hacky, but it works.
if os.path.exists("environment_cpu.yml"):
    with open("environment_cpu.yml") as f:
        conda_environment_cpu = f.read().splitlines()
        conda_environment_cpu = set(
            [
                r.strip().replace("- ", "").split("#")[0].strip().replace("==", "=")
                for r in conda_environment_cpu
                if r and not r.strip().startswith("#") and " - " in r
            ]
        )

    with open("pyproject.toml") as f:
        contents = f.read()

        # Get between dependencies = [ and ]
        start = contents.index("dependencies = [") + len("dependencies = [")
        end = contents.index("]", start)
        pip_pyproject_toml = contents[start:end].splitlines()
        pip_pyproject_toml = set(
            [
                r.strip().strip('"').strip("'").replace('",', "").replace("==", "=")
                for r in pip_pyproject_toml
                if r.strip() and not r.strip().startswith("#")
            ]
        )

        # Add between dev = [ and ]
        start = contents.index("dev = [") + len("dev = [")
        end = contents.index("]", start)
        pip_pyproject_toml_dev = contents[start:end].splitlines()
        pip_pyproject_toml_dev = set(
            [
                r.strip().strip('"').strip("'").replace('",', "").replace("==", "=")
                for r in pip_pyproject_toml_dev
                if r.strip() and not r.strip().startswith("#")
            ]
        )
        pip_pyproject_toml = pip_pyproject_toml.union(pip_pyproject_toml_dev)

    for one_that_can_differ in [
        "python=3.9",
        "conda-forge",
        "pip:",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "nodefaults",
    ]:
        if one_that_can_differ in conda_environment_cpu:
            conda_environment_cpu.remove(one_that_can_differ)
        if one_that_can_differ in pip_pyproject_toml:
            pip_pyproject_toml.remove(one_that_can_differ)

    # Remove entries that are identical in both
    for entry in conda_environment_cpu.intersection(pip_pyproject_toml):
        conda_environment_cpu.remove(entry)
        pip_pyproject_toml.remove(entry)

    assert len(conda_environment_cpu) == 0, "Differences between conda and pip requirements:\n" + "\n".join(
        conda_environment_cpu
    )
    assert len(pip_pyproject_toml) == 0, "Differences between conda and pip requirements:\n" + "\n".join(
        pip_pyproject_toml
    )
setup()

# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This script defines and runs profiling benchmarks for different versions of the `qdecomp` package.
It creates temporary virtual environments for each version, installs the package, and runs the
profiling script.
"""

from typing import List

import os
import tempfile
import venv
import subprocess
import shutil

from benchmark_utils import get_package_versions


def filter_versions(versions: List[str]) -> List[str]:
    """
    Filters a list of version identifiers, returning only those that meet the specified criteria for testing.
    The specific filtering logic should be implemented as needed, for example, to include only versions greater than or equal to a certain threshold.

    Parameters:
        versions (list): A list of version identifiers to be evaluated.

    Returns:
        list: A filtered list containing only the versions that satisfy the selection criteria.
    """
    # Filtering logic here, e.g., only versions >= 1.0.0
    return versions

def run_single_benchmark(version: str) -> None:
    """
    Runs the profiling of the package for the given version.

    To profile a specific version, this function creates a temporary virtual environment, installs
    the specified version of the package, copies the profiling scripts (`benchmarking_functions.py`)
    in the virtual environment and runs the profiling script within that environment. Copying the
    profiling script ensures that the latest version of the script is used, as it may change between
    package versions.

    Args:
        version (str): The version of the package to profile.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_dir = os.path.join(tmpdir, "venv")
        venv.create(venv_dir, with_pip=True)
        if os.name != "nt":
            python_bin = os.path.join(venv_dir, "bin", "python")
        else:
            python_bin = os.path.join(venv_dir, "Scripts", "python.exe")

        venv_script_path = os.path.join(tmpdir, "benchmarking_functions.py")
        script_to_copy_path = os.path.join(os.path.dirname(__file__), "benchmarking_functions.py")
        data_path = os.path.join(os.path.dirname(__file__))

        # Install specific package versions and run the profiling script.
        try:
            subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([python_bin, "-m", "pip", "install", f"qdecomp=={version}"])

            # Copy the benchmarking script into the virtual environment.
            # This ensures the latest version of benchmarking_functions.py is used (it might change from one version to another).
            shutil.copyfile(script_to_copy_path, venv_script_path)

            subprocess.check_call([python_bin, venv_script_path, "--data_path", data_path], text=True)

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing version {version}: {e}")
            print(f"Return code: {e.returncode}")
            print(f"Command: {e.cmd}")

def run_all_benchmarks(versions: List[str] = None, rerun: bool = False) -> None:
    """
    Runs the profiling for all relevant package versions.

    Args:
        versions (list): A list of versions to profile. If `None`, profiles versions obtained from `get_package_versions()` after filtering.
        rerun (bool): If `True`, re-runs profiling even if profiles already exist. Default is `False`.
    """
    if versions is None:
        versions = get_package_versions()
        versions = filter_versions(versions)

    if not rerun:
        benchmarks_dir = os.path.join(os.path.dirname(__file__), "data")
        if os.path.exists(benchmarks_dir):
            benchmarked_versions = os.listdir(benchmarks_dir)
            versions = [v for v in versions if "v" + v.replace('.', '_') not in benchmarked_versions]

    for v in versions:
        run_single_benchmark(v)


if __name__ == "__main__":
    # run_single_benchmark("1.0.1")
    run_all_benchmarks(versions=None, rerun=False)

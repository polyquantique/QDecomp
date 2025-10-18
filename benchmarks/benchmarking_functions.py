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
This script is copied by benchmarks/run_benchmarks.py to be run inside a virtual environment.
It performs the profiling of the QDecomp package and saves the profiling data to the specified path.
"""

import argparse
import cProfile
import os
from time import perf_counter

import numpy as np

import qdecomp
from qdecomp.decompositions import rz_decomp

# Define benchmark parameters
ANGLE_LIST = np.array([1 / 3, 7 / 12, 13 / 12]) * np.pi
EPSILON_LIST = 10.0 ** np.arange(-1, -7.5, -1)


def run_single_profile(func: callable, args: list = [], kwargs: dict = {}) -> cProfile.Profile:
    """
    Runs a single profiling session for the given function with the specified arguments.

    Args:
        func (callable): The function to profile.
        args (list): Positional arguments to pass to the function.
        kwargs (dict): Keyword arguments to pass to the function.

    Returns:
        cProfile.Profile: The profiling data collected during the function execution.
    """
    profiler = cProfile.Profile(timer=perf_counter)
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    return profiler


def save_profile(
    data_path: str,
    profile: cProfile.Profile,
    benchmark_name: str,
    args: list = [],
    kwargs: dict = {},
) -> None:
    """
    Saves the profiling data to a file in the specified data path. The file name is constructed
    based on the benchmark name and the arguments used.

    Args:
        data_path (str): The base path to save profiling data.
        profile (cProfile.Profile): The profiling data to save.
        benchmark_name (str): The name of the benchmark used in the file name.
        args (list): Positional arguments used in the benchmark.
        kwargs (dict): Keyword arguments used in the benchmark.
    """
    base_path = os.path.join(data_path, "data")
    version_name = "v" + qdecomp.__version__.replace(".", "_")
    file_dir = os.path.join(base_path, version_name)
    os.makedirs(file_dir, exist_ok=True)

    args_name = "-".join([str(a) for a in args])
    kwargs_name = "-".join([key + str(val) for key, val in kwargs.items()])

    file_name = "-".join([benchmark_name, args_name, kwargs_name])
    file_name = file_name.rstrip("-").replace(".", "_").replace("--", "-")

    profile.dump_stats(os.path.join(file_dir, file_name + ".prof"))


def profile_package(data_path: str) -> None:
    """
    Performs the profiling of the QDecomp package.

    Args:
        data_path (str): Path to save profiling data.
    """
    n_profiles = len(ANGLE_LIST) * len(EPSILON_LIST)
    iteration = 1
    for a in ANGLE_LIST:
        for e in EPSILON_LIST:
            print(f"Running profile {iteration} / {n_profiles}...", end="\r")
            kwargs = {"angle": a, "epsilon": e}
            profile = run_single_profile(rz_decomp, kwargs=kwargs)
            save_profile(data_path, profile, "rz_decomp", kwargs=kwargs)
            iteration += 1
    print(f"Profiling of version {qdecomp.__version__} completed.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to save profiling data")
    args = parser.parse_args()

    # Profile the package
    profile_package(args.data_path)

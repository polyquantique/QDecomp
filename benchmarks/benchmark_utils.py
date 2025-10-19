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
This file contains utility functions for benchmarking the `qdecomp` package.
"""

import json
import os
import pstats
import re
import urllib.request
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gather_machine_info() -> None:
    """
    Collect and save information about the machine running the benchmarks.
    """
    print(
        "Benchmarks performance depends on the machine used. Please provide the following information:"
    )
    os_name = input("Operating System: ")
    arch = input("Architecture (e.g. x64): ")
    cpu = input("CPU (e.g. 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz): ")
    num_cpu = input("Number of CPUs: ")
    ram = input("RAM (e.g. 16GB): ")

    machine_info = {"os": os_name, "arch": arch, "cpu": cpu, "num_cpu": num_cpu, "ram": ram}

    json_path = os.path.join(os.path.dirname(__file__), "machine.json")
    json.dump(machine_info, open(json_path, "w"), indent=4)


def load_machine_info(as_dict=True) -> Union[dict, str]:
    """
    Load and return information about the machine running the benchmarks. If `as_dict` is `True`,
    returns the information as a dictionary. Otherwise, the information is returned as a string.

    Returns:
        dict: A dictionary containing machine information such as OS, architecture, CPU, number of CPUs, and RAM.
    """
    json_path = os.path.join(os.path.dirname(__file__), "machine.json")
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if as_dict:
        return info

    else:
        info_str = (
            f"Operating System: {info['os']}\n"
            f"Architecture: {info['arch']}\n"
            f"CPU: {info['cpu']}\n"
            f"Number of CPUs: {info['num_cpu']}\n"
            f"RAM: {info['ram']}\n"
        )
        return info_str


def get_package_versions() -> list[str]:
    """
    Retrieves all available versions of the `qdecomp` package from PyPI.

    Returns:
        list: A sorted list of version strings available for the `qdecomp` package.
    """
    url = f"https://pypi.org/pypi/qdecomp/json"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)

    return sorted(data["releases"].keys())


def load_profile() -> pd.DataFrame:
    """
    Loads profiling data from the `data` directory, extracting relevant statistics for analysis.

    Returns:
        pd.DataFrame: A DataFrame containing profiling data across different versions, angles, and epsilon values.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data")
    versions = os.listdir(data_path)
    data = pd.DataFrame(columns=["version", "angle", "epsilon", "cum_time"])

    for v in versions:
        version_path = os.path.join(data_path, v)
        for profile in os.listdir(version_path):
            if "rz_decomp" in profile:
                stats = pstats.Stats(os.path.join(version_path, profile))

                for key, val in stats.stats.items():
                    # key: (filename, line_number, function_name)
                    # val: (ncalls, nprimitivecalls, tottime, cumtime, callers)
                    if "rz_decomp" in key[2]:
                        version = v.replace("_", ".")
                        epsilon = (
                            re.search(r"epsilon(0_[0-9]+|1e-[0-9]+)", profile)
                            .group(1)
                            .replace("_", ".")
                        )
                        angle = re.search(r"angle([0-9_]+)", profile).group(1).replace("_", ".")
                        cum_time = val[3]

                        new_row = {
                            "version": version,
                            "angle": float(angle),
                            "epsilon": float(epsilon),
                            "cum_time": float(cum_time),
                        }
                        data.loc[len(data)] = new_row

    return data


def plot_version_profiling() -> None:
    """
    Plots the profiling results across different package versions and saves the graph as an SVG file
    in the `graphs` directory.
    """
    data = load_profile()

    version_list = sorted(data["version"].unique(), key=lambda x: list(map(int, x[1:].split("."))))
    epsilon_list = sorted(data["epsilon"].unique(), reverse=True)

    plt.figure(figsize=(6, 4))

    for e in epsilon_list:
        time_list = [
            data[(data["version"] == v) & (data["epsilon"] == e)]["cum_time"].mean()
            for v in version_list
        ]

        plt.semilogy(version_list, time_list, marker="o", label=f"$\\varepsilon$ = {e}")

    plt.xlabel("Version")
    plt.ylabel("Average Runtime (s)")
    plt.title("rz_decomp() Profiling Across Versions")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    plt.savefig(os.path.join(graphs_dir, "version_profiling.svg"))


def plot_epsilon_profiling() -> None:
    """
    Plots the profiling results for different epsilon values and saves the graph as an SVG file
    in the `graphs` directory. The latest package version is used for this plot.
    """
    data = load_profile()

    version = max(data["version"].unique(), key=lambda x: list(map(int, x[1:].split("."))))
    angle_list = sorted(data["angle"].unique())
    epsilon_list = sorted(data["epsilon"].unique(), reverse=True)
    inv_epsilon_list = [1 / e for e in epsilon_list]

    plt.figure(figsize=(6, 4))

    for a in angle_list:
        time_list = [
            data[(data["version"] == version) & (data["angle"] == a) & (data["epsilon"] == e)][
                "cum_time"
            ].mean()
            for e in epsilon_list
        ]

        plt.loglog(
            inv_epsilon_list, time_list, marker="o", label=f"$\\theta$ = {a/np.pi:.2f}$\\pi$"
        )

    plt.xlabel("Inverse of the Decomposition Tolerance ($\\varepsilon^{-1}$)")
    plt.ylabel("Runtime (s)")
    plt.title(f"{version}: rz_decomp() Runtime VS Decomposition Tolerance")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    plt.savefig(os.path.join(graphs_dir, "epsilon_profiling.svg"))


if __name__ == "__main__":
    plot_version_profiling()
    plot_epsilon_profiling()

import json
import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import re
import pstats
import numpy as np


def machine_info():
    print("Benchmarks performance depends on the machine used. Please provide the following information:")
    os_name = input("Operating System: ")
    arch = input("Architecture (e.g. x64): ")
    cpu = input("CPU (e.g. 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz): ")
    num_cpu = input("Number of CPUs: ")
    ram = input("RAM (e.g. 16GB): ")

    machine_info = {
        "os": os_name,
        "arch": arch,
        "cpu": cpu,
        "num_cpu": num_cpu,
        "ram": ram
    }

    json_path = os.path.join(os.path.dirname(__file__), 'machine.json')
    json.dump(machine_info, open(json_path, 'w'), indent=4)

def get_package_versions():
    url = f"https://pypi.org/pypi/qdecomp/json"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)

    return sorted(data["releases"].keys())

def load_profile():
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
                        epsilon = re.search(r"epsilon([0-9_]+)", profile).group(1).replace("_", ".")
                        angle = re.search(r"angle([0-9_]+)", profile).group(1).replace("_", ".")
                        cum_time = val[3]

                        new_row = {"version": version, "angle": float(angle), "epsilon": float(epsilon), "cum_time": float(cum_time)}
                        data.loc[len(data)] = new_row
    
    return data

def plot_version_profiling():
    data = load_profile()
    
    version_list = sorted(data["version"].unique(), key=lambda x: list(map(int, x[1:].split('.'))))
    epsilon_list = sorted(data["epsilon"].unique(), reverse=True)

    plt.figure(figsize=(6, 4))

    for e in epsilon_list:
        time_list = [data[(data["version"] == v) & (data["epsilon"] == e)]["cum_time"].mean() for v in version_list]
        
        plt.plot(version_list, time_list, marker="o", label=f"$\\varepsilon$ = {e}")    

    plt.xlabel("Version")
    plt.ylabel("Average Runtime (s)")
    plt.title("rz_decomp() Profiling Across Versions")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "graphs", "version_profiling.svg"))

def plot_epsilon_profiling():
    data = load_profile()

    version = max(data["version"].unique(), key=lambda x: list(map(int, x[1:].split('.'))))
    angle_list = sorted(data["angle"].unique())
    epsilon_list = sorted(data["epsilon"].unique(), reverse=True)
    inv_epsilon_list = [1/e for e in epsilon_list]

    plt.figure(figsize=(6, 4))

    for a in angle_list:
        time_list = [data[(data["version"] == version) & (data["angle"] == a) & (data["epsilon"] == e)]["cum_time"].mean() for e in epsilon_list]

        plt.semilogx(inv_epsilon_list, time_list, marker="o", label=f"$\\theta$ = {a/np.pi:.2f}$\\pi$")

    plt.xlabel("Inverse of the Decomposition Tolerance ($\\varepsilon^{-1}$)")
    plt.ylabel("Runtime (s)")
    plt.title(f"{version}: rz_decomp() Runtime VS Decomposition Tolerance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "graphs", "epsilon_profiling.svg"))


if __name__ == "__main__":
    plot_version_profiling()
    plot_epsilon_profiling()

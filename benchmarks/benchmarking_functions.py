import cProfile
from qdecomp.decompositions import rz_decomp
import qdecomp
import numpy as np
import os
from time import perf_counter
import argparse


ANGLE_LIST = np.array([1/3, 7/12, 13/12]) * np.pi
EPSILON_LIST = 10.0 ** np.arange(-1, -3.5, -1)


def run_single_profile(func, args=[], kwargs={}):
    profiler = cProfile.Profile(timer=perf_counter)
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    
    return profiler

def save_profile(data_path, profile, benchmark_name, args=[], kwargs={}):
    base_path = os.path.join(data_path, "data")
    version_name = "v" + qdecomp.__version__.replace('.', '_')
    file_dir = os.path.join(base_path, version_name)
    os.makedirs(file_dir, exist_ok=True)

    args_name = "-".join([str(a) for a in args])
    kwargs_name = "-".join([key+str(val) for key, val in kwargs.items()])

    file_name = "-".join([benchmark_name, args_name, kwargs_name])
    file_name = file_name.rstrip("-").replace(".", "_").replace("--", "-")

    profile.dump_stats(os.path.join(file_dir, file_name + '.prof'))

def profile_package(data_path):
    """
    Performs the profiling of the QDecomp package.
    """
    for a in ANGLE_LIST:
        for e in EPSILON_LIST:
            kwargs = {'angle': a, 'epsilon': e}
            profile = run_single_profile(rz_decomp, kwargs=kwargs)
            save_profile(data_path, profile, "rz_decomp", kwargs=kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to save profiling data")
    args = parser.parse_args()

    profile_package(args.data_path)

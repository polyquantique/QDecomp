import cProfile
import os
import tempfile
import venv
import subprocess
import shutil

from benchmark_utils import get_package_versions


def filter_versions(versions):
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

def run_single_benchmark(version):
    """
    Runs the profiling of the package for the given version.
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

def run_all_benchmarks(rerun=False, versions=None):
    """
    Runs the profiling for all relevant package versions.

    Args:
        rerun (bool): If `True`, re-runs profiling even if profiles already exist. Default is `False`.
        specific_versions (list): A list of specific package versions to profile. If `None`, profiles all versions.
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
    run_all_benchmarks(rerun=False, versions=None)

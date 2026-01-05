# Script to install marvel conda environment
# python setup_conda_env.py

import subprocess
import sys
import os

def run_command(command):
    """
    Run a command using subprocess and handle exceptions.
    """
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Command executed successfully: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {' '.join(command)}")
        print(e)
        sys.exit(1)

def check_conda_installed():
    """
    Check if Conda is installed by trying to get the version.
    """
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conda is installed.")
    except FileNotFoundError:
        print("Conda is not installed. Please install Conda before running this script.")
        sys.exit(1)

def create_conda_env():
    """
    Create a Conda environment from the environment.yml file.
    """
    env_name = "marvel"
    user_home = os.path.expanduser("~")  # This gets the home directory path in a cross-platform way
    env_path = os.path.join(user_home, "miniconda", "envs", env_name)
    print(f"Creating environment at: {env_path}")
    run_command(["conda", "env", "create", "-f", "environment.yml", "--prefix", env_path])

def main():
    # Check if Conda is installed
    check_conda_installed()

    # Create the environment from environment.yml
    create_conda_env()

    print("Environment has been successfully created and all packages installed.")

if __name__ == "__main__":
    main()


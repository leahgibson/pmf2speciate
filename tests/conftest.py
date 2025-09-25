import sys
import os

# This is a standard way to add the src directory to the Python path
# for a pytest run. It ensures that the tests can find the 'pmf2speciate'
# package, even when running 'pytest' from the project root.

# Get the directory of the conftest.py file
test_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (one level up from the tests directory)
project_root = os.path.join(test_dir, "..")

# Get the src directory
src_dir = os.path.join(project_root, "src")

# Insert the src directory at the beginning of the Python path
# This ensures it's found before the current working directory.
sys.path.insert(0, src_dir)

print(f"Added {src_dir} to Python path for testing.")

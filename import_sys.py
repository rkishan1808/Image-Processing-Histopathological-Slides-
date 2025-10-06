import sys
import os

# Correct the path to where your module is located
module_directory = '/Users/ravikishan/my_project/helpers'

# Append the module directory to sys.path if it exists
if os.path.exists(module_directory):
    sys.path.append(module_directory)
else:
    print(f"Directory does not exist: {module_directory}")

# Print sys.path to confirm the directory is added
print(sys.path)

# Import the module
try:
    import helper_functions
    print("Module imported successfully")
except ModuleNotFoundError as e:
    print(f"Error importing module: {e}")
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
clone_path = os.getenv("CLONE_TARGET_PATH", "./clone")
full_path = os.path.abspath(os.path.join(base_dir, clone_path))
print(full_path)

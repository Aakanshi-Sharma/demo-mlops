import os

dirs=[
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "src",
    "saved_models",
    "notebooks"
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, ".gitkeep"), "w") as f:
        pass

files=[
    "dvc.yaml",
    "params.yaml",
    os.path.join("src","__init__.py"),
    ".gitignore"
]

for file in files:
    with open(file, "w") as f:
        pass
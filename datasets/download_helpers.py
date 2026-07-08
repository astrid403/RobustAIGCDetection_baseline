from pathlib import Path


def require_directory(path, message):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{message}\nChecked path: {path}")
    return path


def print_missing_dataset(dataset, path, hint):
    print(f"{dataset} folder not found at {path}.")
    print(hint)


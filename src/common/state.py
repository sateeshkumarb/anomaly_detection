from pathlib import Path


def get_root_directory() -> str:
    cwd = Path(__file__).resolve().parent
    return f"{cwd.parent.parent}/saved_states"

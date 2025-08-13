from pathlib import Path
def get_root_directory() -> Path:
    """Get the root directory (two levels up from this script's location)."""
    return Path(__file__).resolve().parents[0]

root_dir = get_root_directory()
        
params_path = root_dir / 'params.yaml'
print(f"Root directory: {get_root_directory()}")
print(f"Looking for params at: {params_path}")
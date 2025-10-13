from importlib import resources
from omegaconf import OmegaConf
import os

def load_config(path: str | None = None):
    if path:
        return OmegaConf.load(path)
    env_path = os.getenv("BIOMEDKAI_CONFIG")
    if env_path:
        return OmegaConf.load(env_path)
    with resources.files("biomedkai.configs").joinpath("default.yaml").open("rb") as f:
        return OmegaConf.load(f)

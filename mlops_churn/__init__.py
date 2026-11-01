"""MLOps investigation agent and independent synthetic benchmark."""
from .scenarios import generate_scenarios

__all__ = ['investigate', 'generate_scenarios']


def __getattr__(name):
    # Generating scenarios only needs the scientific dependencies.
    if name == 'investigate':
        from .agent import investigate
        return investigate
    raise AttributeError(name)

"""LIME image explanations and coarse-to-fine budget allocation.

The baseline LIME implementation is adapted from Marco Tulio Ribeiro's LIME
project. See LICENSE.txt and the source references in README.md.
"""

from .coarse_to_fine import explain_instance_coarse_to_fine
from .lime_image import ImageExplanation, LimeImageExplainer

__all__ = [
    "ImageExplanation",
    "LimeImageExplainer",
    "explain_instance_coarse_to_fine",
]

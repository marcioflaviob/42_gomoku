from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("ai.heuristics", ["ai/heuristics.pyx"], include_dirs=[np.get_include()]),
    Extension("ai.moves", ["ai/moves.pyx"], include_dirs=[np.get_include()]),
    Extension("ai.minimax", ["ai/minimax.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="gomoku-ai-cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)

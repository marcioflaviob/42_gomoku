from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("ai_cython.constants", ["ai_cython/constants.pyx"]),
    Extension("ai_cython.heuristics", ["ai_cython/heuristics.pyx"], include_dirs=[np.get_include()]),
    Extension("ai_cython.moves", ["ai_cython/moves.pyx"], include_dirs=[np.get_include()]),
    Extension("ai_cython.optimizer", ["ai_cython/optimizer.pyx"], include_dirs=[np.get_include()]),
    Extension("ai_cython.minimax", ["ai_cython/minimax.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="gomoku-ai-cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)

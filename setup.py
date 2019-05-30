# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

def main():
    setup(
        name             = 'iscat_lib',
        version          = '0.1.0',
        packages         = find_packages(),
        author           = 'Mariia Dmitrieva, Joel Lefebvre',
        author_email     = 'mariia.dmitrieva@eng.ox.ac.uk, joel.lefebvre@eng.ox.ac.uk',
        url              = "https://github.com/MariiaVision/iSCAT_tracking.git",
        description      = 'Particle tracker for iSCAT data (collaboration with Eggeling group)',
        install_requires = ['setuptools', 'wheel', 'numpy', 'scipy', 'scikit-image',
                            'matplotlib', 'tqdm'],
    )

if __name__ == "__main__":
    main()

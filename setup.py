import setuptools

with open("README_pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name             = 'trait2d',
    description      = 'Cross-platform Python software package and GUIs to support Single Particle Tracking experiments.',
    long_description =long_description,
    long_description_content_type="text/markdown",
    author           = 'Francesco Reina, John Wigg, Mariia Dmitrieva, Bela Vogler, Joël Lefebvre,',
    author_email     = 'francesco.reina@uni-jena.de, john.wigg@uni-jena.de, mariia.dmitrieva@eng.ox.ac.uk, bela.vogler@uni-jena.de, lefebvre.joel@uqam.ca',
    url              = "https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D",
    packages         = setuptools.find_packages(),
    version          = "1.2",
    install_requires = ['setuptools', 'wheel', 'numpy', 'scipy', 'scikit-image',
                        'matplotlib', 'opencv-python', 'tqdm', 'pandas',
                        'tk', 'imageio', 'PyQt5', 'pyqtgraph'],
    entry_points     = {
                       'console_scripts': [
                            'trait2d_analysis_gui = trait2d_analysis_gui:main',
                            'trait2d_simulator_gui = trait2d_simulator_gui:main',
                            'trait2d_tracker_gui = trait2d_tracker_gui:main'
                        ],
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>=3.7',
    include_package_data = True
)

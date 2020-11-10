import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name             = 'trait2d-john-wigg',
    description      = 'Cross-platform Python software package and GUIs to support Single Particle Tracking experiments.',
    long_description =long_description,
    long_description_content_type="text/markdown",
    author           = 'Mariia Dmitrieva, Joel Lefebvre, Francesco Reina, John Wigg',
    author_email     = 'mariia.dmitrieva@eng.ox.ac.uk, joel.lefebvre@eng.ox.ac.uk, francesco.reina@uni-jena.de, john.wigg@uni-jena.de',
    url              = "https://github.com/FReina/TRAIT-2D",
    packages         = setuptools.find_packages(),
    version_config={
        "version_format": "{tag}.dev-{sha}",
        "starting_version": "0.1.0"
    },
    setup_requires=['better-setuptools-git-version'],
    install_requires = ['setuptools', 'wheel', 'numpy', 'scipy', 'scikit-image',
                        'matplotlib', 'opencv-python', 'tqdm', 'pandas',
                        'tk', 'imageio', 'PyQt5', 'pyqtgraph'],
    entry_points     = {
                       'console_scripts': [
                            'trait_analysis_gui = gui_analysis:main',
                            'trait_simulator_gui = gui_simulator:main',
                            'trait_tracker_gui = gui_tracker:main'
                        ],
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>=3.7',
)

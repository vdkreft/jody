import setuptools

setuptools.setup(
    name="jody",
    version="1.0",
    author="Klaus Pontoppidan",
    author_email="klaus.m.pontoppidan@jpl.nasa.gov",
    description="A package to calculate icy opacities using optool",
    packages=['jody','examples'],
    package_data={'jody': ['ocs/*.lnk','ocs/*.asc']},
    install_requires=['optool'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

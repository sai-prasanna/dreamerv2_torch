import setuptools
import pathlib


setuptools.setup(
    name="dreamerv2",
    version="2.2.0",
    description="Mastering Atari with Discrete World Models - Pytorch",
    url="http://github.com/sai-prasanna/dreamerv2",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["dreamerv2_torch", "dreamerv2_torch.common"],
    package_data={"dreamerv2_torch": ["configs.yaml"]},
    entry_points={"console_scripts": ["dreamerv2_torch=dreamerv2_torch.train:main"]},
    install_requires=[
        "gym[atari]",
        "atari_py",
        "crafter",
        "dm_control",
        "ruamel.yaml",
        "torch",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

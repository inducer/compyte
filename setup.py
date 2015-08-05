from setuptools import setup

setup(
    name = "compyte",
    version = "0.0.1",
    author = "Andreas Kloeckner",
    author_email = "inform@tiker.net",
    description = "A common set of compute primitives for PyCUDA and PyOpenCL",
    license = "MIT",
    keywords = "pyopencl pycuda",
    url = "https://github.com/inducer/compyte",
    packages=["compyte"],
    install_requires=["numpy", "pytools"],
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)

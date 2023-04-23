from setuptools import setup, find_packages

setup(
    name='faculai',
    version='1.0.0',
    description = "A Python library for detecting and analyzing solar faculae in HMI images using machine learning.",
    url = "https://github.com/Rawkul/faculai",
    author = "Antonio Reche García",
    author_email = "antonioreche.ds@gmail.com",
    license = "MIT",
    packages=['faculai'],
    install_requires = [
      "tensorflow==2.10",
      "keras",
      "scipy",
      "numpy",
      "pandas"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    include_package_data=True
)

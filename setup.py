from setuptools import setup

setup(
    name='faculai',
    version='2.0.1',
    description = "A Python library for detecting and analyzing solar faculae in HMI images using machine learning.",
    url = "https://github.com/rawkul/faculai",
    author = "Antonio Reche García",
    author_email = "antonioreche.ds@gmail.com",
    license = "MIT",
    packages=['faculai'],
    install_requires = [
      "tensorflow>=2.11.1,<2.12", # Force version not to be 2.12 or higher.
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

# Examples for a Tutorial on Reproducible Environments

This repository contains examples to be used for a tutorial on reproducible environments. Instructions to run the examples are provided below.

## Requirements to run the examples

### Python

You will need access to a Python 3 intepreter that you can run on a command line interface. This means the `python3` command should run successfully on your preferred command line interface.

Additionally, you will need access to the `venv` Python module. If you are on a macOS system and installed Python via Homebrew, this module should already be available to you.
However, on an Ubuntu Linux system (including Ubuntu via Windows Subsystem for Linux), you will need to install this capability via the APT package manager. Assuming you are on an Ubuntu Linux system where you have `sudo` access, run:
```
sudo apt install python3-venv
```

### Docker

You will need to install Docker Desktop for your system.

## Python example with `venv` and Docker

The `python-example` directory contains a short Python script named `ilu_study.py` that computes a matrix factorization. The code depends on the NumPy and SciPy libraries. The are two options for setting up a reproducible environment to run the code.

### `venv`

Create a Python virtual environment where you will install the script's dependencies by running the following command on your command line interface:
```
python3 -m venv repro-venv
```
Activate the environment:
```
source repro-venv/bin/activate
```
At this point, you should confirm that the Python interpreter and `pip` are coming from your new virtual environment by running `which python` and `which pip`. Once you have confirmed this, install the dependencies by running:
```
pip install numpy scipy
```
Then, navigate to the `python-example` directory and run the code (with default options):
```
python ilu_study.py
```
The `python-example` directory also provides a `requirements.txt` file which you can alternatively use to install the dependencies as follows:
```
pip install -r requirements.txt
```
Note that the NumPy and SciPy versions listed in this `requirements.txt` file were pulled from a virtual environment based on Python 3.12.3. If your Python version is older than this, it is possible the specific NumPy and SciPy versions
are not compatible with your Python installation.

### Docker

The `python-example` directory contains a Dockerfile which you can use to build a Docker image. To build the image, navigate to the `python-example` directory and run:
```
docker image build -t ilu-study .
```
Then, run a container based on the image:
```
docker container run --rm ilu-study python ilu_study.py
```

## TensorFlow example with Docker

The `tensorflow-example` directory contains a script named `classify_imagenet_resnet.py`, which classifies an image that is also provided in the directory. Credit for the code in this script goes to the Keras documentation page ["Usage examples for image classification models"](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

To run this example, first open Docker Desktop to ensure Docker is running. Then, pull the latest TensorFlow CPU Docker image by running the following command on your command line interface:
```
docker pull tensorflow/tensorflow
```
Then, navigate to the `tensorflow-example` directory and run:
```
docker container run --rm --mount type=bind,src=.,target=/home --workdir=/home tensorflow/tensorflow python classify_imagenet_resnet.py

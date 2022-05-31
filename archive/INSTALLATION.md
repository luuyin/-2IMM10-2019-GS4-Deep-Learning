General Requirements:
* Python 3.5.4 (installed via miniconda3)
* Tensorflow 1.3.0 (installed via miniconda3)
* Keras 2.1.2 (**latest version from keras github**)

**<< IMPORTANT >>** 

Notice that Keras 2.0.6 is default installation of miniconda3. This will cause error when loading stored model. To upgrade keras version, please follow above instruction. And make sure it is ran under your environment.

* Jupyter notebook with python3 kernel (see the configuration if you only have python2 kernel)


## Installation (For Linux) - and applicable for user of grid cluster environment 
(in grid cluster environment, packages will be installed under your user account or $HOME directory)

## Install python3 packages via miniconda3
* Link to miniconda site: https://conda.io/miniconda.html
* wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
* chmod +x Miniconda3-latest-Linux-x86_64.sh
* ./Miniconda3-latest-Linux-x86_64.sh

## Create and activate conda virtual environment
* conda create -n tfenv
* source activate tfenv            

## Install tensorflow library

Add conda-forge repository
* conda config --add channels conda-forge

If you have GPU card in your computer:

* conda install tensorflow-gpu

Otherwise:

* conda install tensorflow

## Install required libraries
* conda install ipython
* conda install scipy
* conda install scikit-learn
* conda install h5py
* conda install matplotlib
* pip install -U nltk

run the following code to download all nltk_data in your $HOME directory
* python -m nltk.downloader -d (your $HOME directory)/nltk_data all

include the following python code when need to retrieve data sets from the downloaded nltk folder
```
nltk.data.path.append('(your $HOME directory)/nltk_data')
```
* conda install gensim

## Install keras
## 1. Via Miniconda

* conda install keras

## 2. From github source 

Make sure you run setup.py under tfenv environment and your miniconda python (instead of default installation of python in server, for instance).

* cd (your-git-directory)
* git clone https://github.com/fchollet/keras.git
* cd keras
* python setup.py install 
pip install git+git://github.com/fchollet/keras.git --upgrade


## Upgrade Keras to the latest version (Keras 2.1.2)

* Check your current version. If it is not the latest version, please upgrade by following instruction as follows
```
import keras
keras.__version__
```

* Make sure you run setup.py under tfenv environment and your miniconda python (instead of default installation of python in server, for instance).

```
pip install git+git://github.com/fchollet/keras.git --upgrade

```

## Configure backend before running keras

## 1. For installation with miniconda3

* Make sure it is done under tfenv environment (source activate tfenv)
* nano $HOME/miniconda3/pkgs/keras-2.0.6-py35_0/etc/conda/activate.d/keras_activate.sh
```
#!/bin/bash
if [ "$(uname)" == "Darwin" ]
then
    # for Mac OSX
    export KERAS_BACKEND=tensorflow
elif [ "$(uname)" == "Linux" ]
then
    # for Linux
    export KERAS_BACKEND=tensorflow
fi

```

## 2. For installation from github source
* nano $HOME/.keras/keras.json
```
{   
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

```

## Update kernel of jupyter notebook to python3

**If you have not installed jupyter yet

* pip install jupyter

**Make sure this is ran in tfenv environment

* pip install ipykernel
* python -m ipykernel install --user

## Calling Jupyter Notebook & allowing Keras run with Tensorflow backend
* Jupyter Notebook --allow-root

### **If you are running experiments on GPU clusters, do not forget to check dependencies of the installed libraries with available cuda and gcc modules

## Activate modules 
* module load cuda/8.0
* module load cudnn/5.1
* module load gcc/5.2.0

## Test Installation
python

```
import tensorflow as tf

graph = tf.constant('Hello world')
session = tf.Session()
print(session.run(graph))
session.close()
```

## Deactivate/quit conda tensorflow environment
source deactivate tfenv



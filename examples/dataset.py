"""
pynet dataset helpers overview
==============================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

First checks
------------

In order to test if the 'pynet' package is installed on your machine, you can
check the package version.
"""

import pynet
print(pynet.__version__)

#############################################################################
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly:

import pynet.configure
print(pynet.configure.info())

#############################################################################
# Import metastasis dataset
# -------------------------
#
# From a simple TSV file, this package provides a common interface to import,
# split and display the decribed dataset:

import pynet
from pynet.dataset import split_dataset
from pynet.dataset import LoadDataset
from pynet.transforms import ZeroPadding, Downsample

dataset_desc = "/neurospin/radiomics_pub/workspace/metastasis_dl/data/dataset.tsv"
kwargs = {
    "load": False,
}
dataset = split_dataset(
    path=dataset_desc,
    dataloader=LoadDataset,
    inputs=["t1"],
    outputs=["mask"],
    label="label",
    transforms=[ZeroPadding(shape=(256, 256, 256)), Downsample(scale=2)],
    test_size=0.05,
    validation_size=0.1,
    number_of_folds=1,
    batch_size=5,
    nb_samples=100,
    verbose=0,
    **kwargs)

#############################################################################
# We have now a test, and multiple folds with train-validation datasets that
# can be used to train our network using cross-validation:


from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from pynet.plotting import plot_data

pprint(dataset)

for batch_data in dataset["test"]:
    print("Inputs: ", batch_data["inputs"].shape)
    print("Outputs: ", batch_data["outputs"].shape)
    print("Labels: ", batch_data["labels"].shape)
    print(dataset["test"].dataset.iloc[0].values)
    plot_data(batch_data["inputs"][:1, :, :, :, 20:100])
    break

plt.show()


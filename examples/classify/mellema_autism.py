"""
pynet: predicting autism
========================

Credit: A Grigis

This practice is based on the IMPAC challenge,
https://paris-saclay-cds.github.io/autism_challenge.

Autism spectrum disorder (ASD) is a severe psychiatric disorder that affects
1 in 166 children. In the IMPAC challenge ML models were trained using the
database's derived anatomical and functional features to diagnose a subject
as autistic or healthy. We propose here to implement the best neural network
to achieve this task and proposed in
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6859452, ie. a dense feedforward
network.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import logging
import pynet
from pynet.datasets import fetch_mellema
from pynet.datasets import DataManager
from pynet.utils import setup_logging
from pynet.plotting import Board, update_board
from pynet.interfaces import DeepLearningInterface
from pynet.metrics import SKMetrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import collections
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

setup_logging(level="info")
logger = logging.getLogger("pynet")

use_toy = False
dtype = "all"

data = fetch_mellema(
    datasetdir="/tmp/mellema",
    mode="train",
    dtype=dtype)


class DenseFeedForwardNet(nn.Module):
    def __init__(self, nb_features):
        """ Initialize the instance.

        Parameters
        ----------
        nb_features: int
            the size of the feature vector.
        """
        super(DenseFeedForwardNet, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(nb_features, 64)),
            ("relu1", nn.LeakyReLU(negative_slope=0.01)),
            ("dropout", nn.Dropout(0.13)),
            ("linear2", nn.Linear(64, 64)),
            ("relu2", nn.LeakyReLU(negative_slope=0.01)),
            ("linear3", nn.Linear(64, 1))
        ]))
        self.layers_alt = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(nb_features, 50)),
            ("relu1", nn.ReLU()),
            ("dropout", nn.Dropout(0.2)),
            ("linear2", nn.Linear(50, 100)),
            ("relu2", nn.PReLU(1)),
            ("linear3", nn.Linear(100, 1))
        ]))

    def forward(self, x):
        return self.layers(x)


def my_loss(x, y):
    logger.debug("Binary cross-entropy loss...")
    device = y.get_device()
    criterion = nn.BCEWithLogitsLoss()
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    y = y.type(torch.float32)
    if device != -1:
        y = y.to(device)
    logger.debug("  x: {0} - {1}".format(x.shape, x.dtype))
    logger.debug("  y: {0} - {1}".format(y.shape, y.dtype))
    return criterion(x, y)


def plot_metric_rank_correlations(metrics):
    """ Display rank correlations for all numerical metrics calculated over N
    experiments.

    Parameters
    ----------
    metrics: DataFrame
        a data frame with all computedd metrics as columns and N rows.
    """
    fig, ax = plt.subplots()
    labels = metrics.columns
    sns.heatmap(spearmanr(metrics)[0], annot=True, cmap=plt.get_cmap("Blues"),
                xticklabels=labels, yticklabels=labels, ax=ax)

def l2_regularizer(signal):
    lambda2 = 0.01
    model = signal.object.model
    all_linear2_params = torch.cat([
        x.view(-1) for x in model.layers[0].parameters()])
    l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
    return l2_regularization

# Condition data to perform the learning task
#
nb_features = data.nb_features
manager = DataManager(
    input_path=data.input_path,
    stratify_label="participants_sex_asd",
    labels=["participants_asd"],
    metadata_path=data.metadata_path,
    number_of_folds=4,
    batch_size=128,
    sampler="random",
    test_size=0.01,
    sample_size=1)

# Build the NN to train
#

## 1) type of network
NN_archi = DenseFeedForwardNet(nb_features)
print(NN_archi)

## 2) use the interface to setup the kind of usage and linked parameter
#     general settings
NN_workable = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1.1e-4,
    metrics=["binary_accuracy", "sk_roc_auc_score"],
    loss=my_loss,
    model=NN_archi)
#
NN_workable.add_observer("regularizer", l2_regularizer)
#     Learning rate adaptatif
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=NN_workable.optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    verbose=True,
    eps=1e-8)
## 3) Add  observer and action
NN_workable.board = Board(port=8097, host="http://localhost", env="main")
NN_workable.add_observer("after_epoch", update_board)




# Now do the train job while estimating results on test
test_history, train_history = NN_workable.training(
    manager=manager,
    nb_epochs=60,
    checkpointdir=None,
    fold_index=0,
    scheduler=scheduler,
    with_validation=(not use_toy))


# Present results of the obtained trained classifier

# reload data in mode test
data = fetch_mellema(
    datasetdir="/tmp/mellema",#"/neurospin/nsap/datasets/impac",
    mode="test",
    dtype=dtype)
x_test = np.load(data.input_path, mmap_mode='r') 
df = pd.read_csv(data.metadata_path, sep="\t")
y_test =  df["participants_asd"].values.squeeze()

manager = DataManager.from_numpy(
        test_inputs=x_test, test_labels=y_test,
        batch_size=10)
    
#~ manager = DataManager(
    #~ input_path=data.input_path,
    #~ labels=["participants_asd"],
    #~ metadata_path=data.metadata_path,
    #~ number_of_folds=2,
    #~ batch_size=10,
    #~ sampler=None,
    #~ test_size=.0001,
    #~ sample_size=1)

# perfrom the prediction (sklearn transform() )
y_pred, X, y_true, loss, values = NN_workable.testing(
    manager=manager,
    with_logit=True,
    logit_function="sigmoid",
    predict=False)
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred", (y_pred.squeeze() > 0.5).astype(int)),
    ("truth", y_true.squeeze()),
    ("prob", y_pred.squeeze())]))

print(result)
fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
cm = SKMetrics("confusion_matrix", with_logit=False)(y_pred, y_true)
sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=ax)
ax.set_xlabel("predicted values")
ax.set_ylabel("actual values")
metrics = {}
sk_metrics = dict(
    (key, val) for key, val in pynet.get_tools()["metrics"].items()
    if key.startswith("sk_"))
for name, metric in sk_metrics.items():
    metric.with_logit = False
    value = metric(y_pred, y_true)
    metrics.setdefault(name, []).append(value)
metrics = pd.DataFrame.from_dict(metrics)
print(classification_report(y_true, y_pred >= 0.4))
print(metrics)
# plot_metric_rank_correlations(metrics)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")

plt.show()


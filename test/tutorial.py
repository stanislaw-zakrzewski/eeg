"""
============================
Tutorial 0: Getting Started
============================

This tutorial takes you through a basic working example of how to use this
codebase, including all the different components, up to the results
generation. If you'd like to know about the statistics and plotting, see the
next tutorial.

"""

# Authors: Vinay Jayaram <vinayjayaram13@gmail.com>
#
# License: BSD (3-clause)


##########################################################################
# Introduction
# --------------------
# To use the codebase you need an evaluation and a paradigm, some algorithms,
# and a list of datasets to run it all on. You can find those in the following
# submodules; detailed tutorials are given for each of them.

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

##########################################################################
# If you would like to specify the logging level when it is running, you can
# use the standard python logging commands through the top-level moabb module
import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004, Beetl2021_A, Beetl2021_B, Dreyer2023, Dreyer2023A, Dreyer2023B, Dreyer2023C, GrosseWentrup2009, Lee2019_MI, Liu2024, Schirrmeister2017, Shin2017A, Stieger2021, Weibo2014, Zhou2016, utils
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance


##########################################################################
# In order to create pipelines within a script, you will likely need at least
# the make_pipeline function. They can also be specified via a .yml file. Here
# we will make a couple pipelines just for convenience


moabb.set_log_level("info")

##############################################################################
# Create pipelines
# ----------------
#
# We create two pipelines: channel-wise log variance followed by LDA, and
# channel-wise log variance followed by a cross-validated SVM (note that a
# cross-validation via scikit-learn cannot be described in a .yml file). For
# later in the process, the pipelines need to be in a dictionary where the key
# is the name of the pipeline and the value is the Pipeline object

pipelines = {}
pipelines["AM+LDA"] = make_pipeline(LogVariance(), LDA())
parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
pipe = make_pipeline(LogVariance(), clf)

pipelines["AM+SVM"] = pipe

##############################################################################
# Datasets
# -----------------
#
# Datasets can be specified in many ways: Each paradigm has a property
# 'datasets' which returns the datasets that are appropriate for that paradigm

print(LeftRightImagery().datasets)

##########################################################################
# Or you can run a search through the available datasets:
print(utils.dataset_search(paradigm="imagery", min_subjects=6))

##########################################################################
# Or you can simply make your own list (which we do here due to computational
# constraints)

# Dreyer2023(), Dreyer2023A(), Dreyer2023B(),Dreyer2023C(),Liu2024()
#
datasets = [Stieger2021()]#, , Weibo2014(), Zhou2016()]

##########################################################################
# Paradigm
# --------------------
#
# Paradigms define the events, epoch time, bandpass, and other preprocessing
# parameters. They have defaults that you can read in the documentation, or you
# can simply set them as we do here. A single paradigm defines a method for
# going from continuous data to trial data of a fixed size. To learn more look
# at the tutorial Exploring Paradigms

fmin = 8
fmax = 35
# You can inject custom scoring directly into the paradigm (single or multi-metric).
# custom_scorer = [
#     accuracy_score,
#     (roc_auc_score, {"needs_threshold": True}),
# ]
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)#, scorer=custom_scorer)

##########################################################################
# Evaluation
# --------------------
#
# An evaluation defines how the training and test sets are chosen. This could
# be cross-validated within a single recording, or across days, or sessions, or
# subjects. This also is the correct place to specify multiple threads.

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=False
)
results = evaluation.process(pipelines)

##########################################################################
# Results are returned as a pandas DataFrame. When multiple metrics are
# provided, MOABB adds a primary `score` plus one column per metric
# (e.g., `score_accuracy_score`, `score_roc_auc_score`).

print(results.head())

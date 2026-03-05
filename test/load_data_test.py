import moabb
from moabb.datasets import Cho2017
from moabb.datasets import PhysionetMI
# CHANGE 1: Import WithinSessionEvaluation instead
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

moabb.set_log_level("info")

# Define pipeline
pipelines = {"LogVar+LDA": make_pipeline(LogVariance(), LDA())}

# Load PhysionetMI
dataset = PhysionetMI()
# dataset.subject_list = dataset.subject_list[:]

# Define Paradigm
paradigm = LeftRightImagery(fmin=8, fmax=35)

# CHANGE 2: Use WithinSessionEvaluation
# This will use 5-fold Cross-Validation inside the single session by default
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset])

results = evaluation.process(pipelines)

print(results.head())
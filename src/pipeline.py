from enum import Enum
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pipeline_components.mvmd import MVMD
from pipeline_components.mvmd_2 import MVMD2

class PipelineType(Enum):
    CSP_LDA = "csp+lda"
    TGSP_SVM = "tgsp+svm"
    MDM = "mdm"
    # Experimental pipelines DO NOT USE
    MVMD_CSP_LDA = "mvmd+csp+lda"
    MVMD2_CSP_LDA = "mvmd2+csp+lda"
    MVMD2_TGSP_SVM = "mvmd2+tgsp+svm"

def get_pipeline(pipeline_type: PipelineType):
    if pipeline_type == PipelineType.CSP_LDA:
        return make_pipeline(CSP(n_components=8), LDA())
    elif pipeline_type == PipelineType.TGSP_SVM:
        return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear"))
    elif pipeline_type == PipelineType.MDM:
        return make_pipeline(Covariances("oas"), MDM(metric="riemann"))
    elif pipeline_type == PipelineType.MVMD_CSP_LDA:
        return make_pipeline(MVMD(K=3,alpha=100000), CSP(n_components=12, reg='shrunk'), LDA())
    elif pipeline_type == PipelineType.MVMD2_CSP_LDA:
        return make_pipeline(MVMD2(K=3,alpha=100000), CSP(n_components=12, reg='shrunk'), LDA())
    elif pipeline_type == PipelineType.MVMD2_TGSP_SVM:
        return make_pipeline(MVMD2(K=3,alpha=100000), Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear"))
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

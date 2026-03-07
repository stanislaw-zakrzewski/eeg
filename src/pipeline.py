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
    pipelines = {
        PipelineType.CSP_LDA: make_pipeline(CSP(n_components=8), LDA()),
        PipelineType.TGSP_SVM: make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")),
        PipelineType.MDM: make_pipeline(Covariances("oas"), MDM(metric="riemann")),
        PipelineType.MVMD_CSP_LDA: make_pipeline(MVMD(K=3,alpha=100000), CSP(n_components=12, reg='shrunk'), LDA()),
        PipelineType.MVMD2_CSP_LDA: make_pipeline(MVMD2(K=3,alpha=100000), CSP(n_components=12, reg='shrunk'), LDA()),
        PipelineType.MVMD2_TGSP_SVM: make_pipeline(MVMD2(K=3,alpha=100000), Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear"))
    }
    
    if pipeline_type not in pipelines:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
    return pipelines[pipeline_type]

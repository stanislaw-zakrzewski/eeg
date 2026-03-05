from moabb.evaluations import CrossSessionEvaluation, CrossSubjectEvaluation, WithinSessionEvaluation

def get_evaluation(paradigm, datasets, evaluation_class=WithinSessionEvaluation, overwrite=False):
    return evaluation_class(
        paradigm=paradigm,
        datasets=datasets,
        overwrite=overwrite,
        hdf5_path=None,
    )

import warnings
import moabb
from data_loading import get_dataset, get_paradigm
from pipeline import get_pipeline, PipelineType
from evaluation import get_evaluation
from visualization import plot_results

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def main():
    bids_dir = 'bids_datasets/sample_dataset'
    dataset = get_dataset(bids_dir)
    paradigm = get_paradigm(channels=['C1', 'C2', 'C5', 'C3', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'Cz'])

    pipelines = {
        PipelineType.CSP_LDA.value: get_pipeline(PipelineType.CSP_LDA),
        PipelineType.TGSP_SVM.value: get_pipeline(PipelineType.TGSP_SVM),
        # Experimental pipelines DO NOT USE
        # PipelineType.MVMD_CSP_LDA.value: get_pipeline(PipelineType.MVMD_CSP_LDA),
        # PipelineType.MVMD2_CSP_LDA.value: get_pipeline(PipelineType.MVMD2_CSP_LDA),
        # PipelineType.MVMD2_TGSP_SVM.value: get_pipeline(PipelineType.MVMD2_TGSP_SVM)
    }

    evaluation = get_evaluation(paradigm, [dataset], overwrite=True)
    results = evaluation.process(pipelines)
    # Save results to csv file, uncomment to save
    # results.to_csv("./results.csv")
    plot_results(results)


if __name__ == "__main__":
    main()

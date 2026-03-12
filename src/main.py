import warnings
import moabb
import json
from data_loading import get_dataset, get_paradigm
from pipeline import get_pipeline, PipelineType
from evaluation import get_evaluation
from visualization import plot_results

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def main():
    with open('src/config.json', 'r') as file:
        config = json.load(file)
    dataset = get_dataset(dataset_path=config['dataset_path'], subject_list=config['subject_list'], interval=config['interval'], paradigm=config['paradigm'])
    paradigm = get_paradigm(channels=config['selected_channels'])

    pipelines = {
        PipelineType.CSP_LDA.value: get_pipeline(PipelineType.CSP_LDA),
        PipelineType.TGSP_SVM.value: get_pipeline(PipelineType.TGSP_SVM),
        # Experimental pipelines DO NOT USE
        # PipelineType.MVMD_CSP_LDA.value: get_pipeline(PipelineType.MVMD_CSP_LDA),
        # PipelineType.MVMD2_CSP_LDA.value: get_pipeline(PipelineType.MVMD2_CSP_LDA),
        # PipelineType.MVMD2_TGSP_SVM.value: get_pipeline(PipelineType.MVMD2_TGSP_SVM)
    }

    evaluation = get_evaluation(paradigm, [dataset], overwrite=config['overwrite'])
    results = evaluation.process(pipelines)
    if config['save_results_to_csv']:
        results.to_csv("./results.csv")
    plot_results(results)
    return results.to_json()


if __name__ == "__main__":
    main()

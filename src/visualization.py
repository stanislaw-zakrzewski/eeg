import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results):
    fig, ax = plt.subplots(figsize=(12, 8))
    results["subj"] = results["subject"].apply(str)
    sns.barplot(
        x="score", y="subj", hue="pipeline", data=results, orient="h", palette="viridis", ax=ax
    )
    ax.set_title("Classification Accuracy")
    plt.show()

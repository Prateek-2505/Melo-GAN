import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_dataset_distribution(csv_path='data/splits/train_split.csv'):
    """
    Visualizes the number of samples per emotion category.
    Inputs: Path to the training split CSV.
    """
    # Load Data (Simulated structure based on your report)
    df = pd.read_csv(csv_path)
    count = df['emotion'].value_counts().to_dict()
    if not count:
        print("No emotion labels found in CSV.")
        return

    # Prepare data for plotting (value_counts is already sorted by count desc)
    emotions = [e.capitalize() for e in count.keys()]
    counts = list(count.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=emotions, y=counts, palette="viridis")
    plt.title('Dataset Distribution by Emotion')
    plt.ylabel('Number of Samples')
    plt.xlabel('Emotion Label')

    # Annotate bars with counts
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()
    

if __name__ == "__main__":
    plot_dataset_distribution()
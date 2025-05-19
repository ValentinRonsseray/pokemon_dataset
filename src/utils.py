from matplotlib.axes import Axes  # Import du type Axis
import pandas as pd
import os

def add_count_labels(ax: Axes, fontsize=10):
    """
    Adds the counts above the bars of a seaborn countplot.

    Parameters:
    - ax: matplotlib axes object returned by sns.countplot
    - fontsize: font size for the count labels
    """
    for p in ax.patches:
        height = int(round(p.get_height()))
        if height > 0:
            ax.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=fontsize)

def load_dataframe(train: bool = False) -> pd.DataFrame:
    """
    Loads the Pokemon dataset from the data directory.

    Args:
        train (bool): If True, filters the dataset to include only training data.

    Returns:
        Dataframe (pd.DataFrame): The loaded Pokemon dataset.
    """
    # Construct the absolute path to the CSV file
    # Assuming this script (utils.py) is in the 'src' directory,
    # and the 'data' directory is at the same level as 'src'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # This should be the project root
    csv_path = os.path.join(project_root, 'data', 'candidate_datasets', 'pokemon_team_rocket_dataset', 'pokemon_team_rocket_dataset.csv')

    try:
        df = pd.read_csv(csv_path)
        if train:
            # Filter the DataFrame to include only rows where 'is_train' is True
            df = df[df['Team Rocket'].notnull()]

        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return None


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    object_columns = df.select_dtypes(include=["object"]).columns

    for column in object_columns:
        df[column] = df[column].astype("category")

    economic_statuses = ["Low", "Middle", "High"]
    df["Economic Status"] = pd.Categorical(
        df["Economic Status"], categories=economic_statuses, ordered=True
    )

    df["Criminal Record"] = df["Criminal Record"].astype("bool")

    return df
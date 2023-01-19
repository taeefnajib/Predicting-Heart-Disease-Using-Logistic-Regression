# Importing all dependencies
import pandas as pd

# Creating the dataframe
def process_dataset(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Cleaning the dataset
    df.drop(["education"], axis=1, inplace=True)
    df.dropna(subset=['cigsPerDay', 'BPMeds', 'BMI'], inplace=True)
    df["totChol"].fillna(df["totChol"].mean(),inplace=True)
    df["heartRate"].fillna(df["heartRate"].mean(),inplace=True)
    df["glucose"].fillna(df["glucose"].mean(), inplace=True)
    df.to_csv("data/processed.csv")
    return df
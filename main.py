# Importing all dependencies 
from sklearn.linear_model import LogisticRegression
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from src.data.make_dataset import process_dataset
from src.data.split_dataset import split_dataset
from src.models.train_model import fit_model

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "data/raw.csv"
    test_size: float = 0.3
    random_state: int = 6

hp = Hyperparameters()

def run_wf(filepath: str, test_size: float, random_state: int) -> LogisticRegression:
    df = process_dataset(filepath)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=test_size, random_state=random_state)
    return fit_model(X_train=X_train, y_train=y_train)


if __name__=="__main__":
    run_wf(filepath=hp.filepath, test_size=hp.test_size, random_state=hp.random_state)
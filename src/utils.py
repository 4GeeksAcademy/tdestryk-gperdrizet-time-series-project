from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import cross_val_score, KFold # type: ignore

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


def cross_val(model, features: pd.DataFrame, labels: pd.Series) -> list:
    '''Reusable helper function to run cross-validation on a model. Takes model,
    Pandas data frame of features and Pandas data series of labels. Returns 
    list of cross-validation fold accuracy scores as percents.'''

    # Define the cross-validation strategy
    cross_validation=KFold(n_splits=7, shuffle=True, random_state=315)

    # Run the cross-validation, collecting the scores
    scores=cross_val_score(
        model,
        features,
        labels,
        cv=cross_validation,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )

    # Make scores positive
    scores=abs(scores)

    # Print mean and standard deviation of the scores
    print(f'Root mean square error: {(scores.mean() * 100):.2f} +/- {(scores.std() * 100):.2f}')

    # Return the scores
    return scores
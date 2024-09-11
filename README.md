# Space Titanic Kaggle Competition - Top 14%

## Overview
This project is part of the Space Titanic Kaggle competition, where the challenge was to predict whether passengers on a simulated space voyage would be transported to an alternate dimension. The approach involved extensive data preprocessing, feature engineering, and optimizing with GridSearchCV a machine learning model using XGBoost.

## Data
The data consisted of various features related to the passengers, including demographic data, cabin details, and expenditures during the voyage. The dataset was split into train and test sets, where the train set was used to build the model and the test set to evaluate its performance.

## Feature Engineering 
For this step i tested the Microsoft Data Wrangler tool from the VSCode extensions tab. The tool helped me save some time with its simple interface and efficient methods. The program starts with the output of DataWrangler, used in this case for gattering as much data as possible from the initial dataset:

```python
def MoreInfo(df):
    # Derive column 'Deck' from column: 'Cabin'
    df.insert(4, "Deck", df["Cabin"].str.split("/").str[0])

    # Derive column 'CabinNumber' from column: 'Cabin'
    df.insert(4, "CabinNumber", df["Cabin"].str.split("/").str[1])

    # Derive column 'Side' from column: 'Cabin'
    df.insert(4, "Side", df["Cabin"].str.split("/").str[-1])

    # Drop column: 'Cabin'
    df = df.drop(columns=['Cabin'])

    # Derive column 'LastName' from column: 'Name'
    df.insert(15, "LastName", df["Name"].str.split(" ").str[-1])

    # Derive column 'FirstName' from column: 'Name'
    df.insert(15, "FirstName", df["Name"].str.split(" ").str[0])

    # Drop column: 'Name'
    df = df.drop(columns=['Name'])

    # Derive column 'Group' from column: 'PassengerId'
    df.insert(1, "Group", df["PassengerId"].str.split("_").str[0])

    # Derive column 'GroupNumber' from column: 'Group'
    df.insert(2, "GroupNumber", df["Group"].str[3:])

    # Drop column: 'Group'
    df = df.drop(columns=['Group'])

    # Derive column 'GroupMemberNumber' from column: 'PassengerId'
    df.insert(1, "GroupMemberNumber", df.apply(lambda row : row["PassengerId"][row["PassengerId"].find("_") + 2:], axis=1))

    # Drop column: 'PassengerId'
    df = df.drop(columns=['PassengerId'])
    return df
```


Key steps in feature engineering included:
- Extracting deck, cabin number, and side from the cabin feature.
- Deriving last name and first name from the passenger's full name.
- Aggregating features like food and service expenditures to determine if passengers were in cryosleep.

## Model Development
The model development process involved:
- Handling missing values with group-based imputation.
- One-hot encoding for categorical variables.
- Hyperparameter tuning of the XGBoost classifier using grid search.

## Results
The final model achieved a validation accuracy that placed the submission in the top 14% of all entries. The successful prediction strategy highlighted the importance of feature engineering and model tuning in predictive accuracy.

## Files
- `train.csv` & `test.csv`: Original data files.
- `dataset.csv`: Combined dataset after initial preprocessing.
- `submission.csv`: Final predictions for the competition.

## Dependencies
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost

## Acknowledgements
Thanks to Kaggle for hosting this engaging competition and to all who have shared helpful kernels and discussions that contributed to the development of this project.

![image](https://github.com/user-attachments/assets/894548d0-c7c4-4d09-b943-ed5b5e41b34e)

# -*- coding: utf-8 -*-
"""

### Different feature selector methods to build an Automatic Feature Selection tool
- Pearson Correlation
- Chi-Square
- RFE
- Embedded
- Tree (Random Forest)
- Tree (Light GBM)

### Dataset: FIFA 19 Player Skills
#### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats

player_df = pd.read_csv("https://raw.githubusercontent.com/pujan08/ML_AutoFeatureSelector/main/fifa19.csv")
print(player_df.info())

numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']

player_df = player_df[numcols+catcols]

traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()

traindf = pd.DataFrame(traindf,columns=features)

y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']

X.head()

len(X.columns)

"""### Set some fixed set of features"""

feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30

"""## Filter Feature Selection - Pearson Correlation

### Pearson Correlation function
"""

def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()

    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]


    # Your code ends here
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, y, num_feats)
print(str(len(cor_feature)), 'selected features')

"""### List the selected features from Pearson Correlation"""

cor_feature

"""## Filter Feature Selection - Chi-Sqaure"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

"""### Chi-Squared Selector function"""

def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.columns[chi_support]
    # Your code ends here
    return chi_support, chi_feature

chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')

"""### List the selected features from Chi-Square"""

chi_feature

"""## Wrapper Feature Selection - Recursive Feature Elimination"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

"""### RFE Selector function"""

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = LogisticRegression()
    rfe_selector = RFE(estimator, n_features_to_select=num_feats, step=1)
    rfe_selector = rfe_selector.fit(X_scaled, y)
    rfe_support = rfe_selector.support_
    rfe_feature = X.columns[rfe_support]
    # Your code ends here
    return rfe_support, rfe_feature

rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')

"""### List the selected features from RFE"""

rfe_feature

"""## Embedded Selection - Lasso: SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = LogisticRegression()
    embedded_lr_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_lr_selector.fit(X_scaled, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.columns[embedded_lr_support]
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')

embedded_lr_feature

"""## Tree based(Random Forest): SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust parameters as needed
    embedded_rf_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_rf_selector.fit(X_scaled, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.columns[embedded_rf_support]
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')

embedded_rf_feature

"""## Tree based(Light GBM): SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    estimator = LGBMClassifier()
    estimator.fit(X, y)
    embedded_lgbm_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.columns[embedded_lgbm_support]
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature

embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')

embedded_lgbm_feature

"""## Putting all of it together: AutoFeatureSelector Tool"""

pd.set_option('display.max_rows', None)
# put all selection together
lengths = [len(cor_support), len(chi_support), len(rfe_support), len(embedded_lr_feature), len(embedded_rf_support), len(embedded_lgbm_support)]

if len(set(lengths)) == 1:
  feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lgbm_feature,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
  feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
  feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
  feature_selection_df.index = range(1, len(feature_selection_df)+1)
  feature_selection_df.head(num_feats)

"""## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?"""

from sklearn.model_selection import train_test_split
def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    dataset = pd.read_csv(dataset_path)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

    # Choose the number of features you want to select (you can adjust this)
    num_feats = len(X.columns)


    # Your code ends here
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)


    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    all_features = {
        'pearson': cor_feature if 'pearson' in methods else [],
        'chi-square': chi_feature if 'chi-square' in methods else [],
        'rfe': rfe_feature if 'rfe' in methods else [],
        'log-reg': embedded_lr_feature if 'log-reg' in methods else [],
        'rf': embedded_rf_feature if 'rf' in methods else [],
        'lgbm': embedded_lgbm_feature if 'lgbm' in methods else [],
    }
    common_features = set(all_features['pearson'])
    for method, features in all_features.items():
        common_features = common_features.intersection(features)

    best_features = list(common_features)
    #### Your Code ends here
    return best_features

best_features = autoFeatureSelector(dataset_path="https://raw.githubusercontent.com/pujan08/ML_AutoFeatureSelector/main/fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features

"""### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features"""


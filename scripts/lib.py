import argparse
import random
import time
import pandas as pd
import numpy as np
import joblib as _joblib

# sklearn models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Other sklearn imports
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score

_UP = '\033[1A'
_CLEAR = '\x1b[2K'
_clfs = {
    'RandomForestClassifier': Pipeline([('scaler', StandardScaler()), ('estimator', RandomForestClassifier(class_weight='balanced'))]),
    'GaussianNB': Pipeline([('scaler', StandardScaler()), ('estimator', GaussianNB())]), 
    'IsolationForest': Pipeline([('scaler', StandardScaler()), ('estimator', IsolationForest())]),
}
_metrics = {
    'f1_score': f1_score,
    'balanced_accuracy_score': balanced_accuracy_score,
}

def summary(data):
    data = pd.read_csv(data) if type(data) == str else data
    summary_df = pd.DataFrame(data.dtypes, columns=['dtypes'])
    summary_df['duplicated'] = data.duplicated().sum()
    summary_df['missing#'] = data.isna().sum()
    summary_df['missing%'] = (data.isna().sum())/len(data)
    summary_df['uniques'] = data.nunique().values
    summary_df['count'] = data.count().values
    return summary_df

def random_stratified_sampling(df, stratify, frac):
    return train_test_split(df, test_size=frac, stratify=df[stratify])[1]

# TODO: Preprocessing ? Scaling ?
def _preprocess(df):
    # Handle NaN values.
    df.dropna(inplace=True) 
    # Encodes categorical columns.
    encoder = OrdinalEncoder() 
    df[df.select_dtypes('object').columns] = encoder.fit_transform(df.select_dtypes('object')) 
    # Deletes columns with 1 unique value.
    df.drop(columns=df.columns[df.nunique()==1], inplace=True)
    return df

def fit(clfs, X, y):
    retval = {}

    for name, clf in clfs.items():
        print(f'Training {name}...')
        t0 = time.time()
        retval[name] = clf.fit(X, y)
        print(_UP, end=_CLEAR)
        print(f'Training {name}: {time.time() - t0}')                

    return retval

def predict(clfs, X):
    retval = []
    
    for name, clf in clfs.items():
        print(f'Predicting {name}...')
        t0 = time.time()
        retval.append(clf.predict(X))
        print(_UP, end=_CLEAR)
        print(f'Predicting {name}: {time.time() - t0}')                
    
    return retval

def _compute_metrics(metrics, y_true, y_preds):
    results = np.zeros((len(y_preds), len(metrics)))
    for i, y_pred in enumerate(y_preds):
        for j, (_, metric) in enumerate(metrics.items()):
            results[i, j] = metric(y_true, y_pred)
    return results

def select_best(data, pipelines, y_cols, metrics, dtype=None, save=True, save_dir='', frac=1):
    data = pd.read_csv(data, dtype=dtype) if type(data) == str else data
    data = _preprocess(data)
    if frac < 1:
        data = random_stratified_sampling(data, y_cols, frac)
    
    X, y = data.drop(columns=y_cols), data[y_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipelines = fit(pipelines, X_train, y_train)
    y_preds = binarize(predict(pipelines, X_test))

    results = _compute_metrics(metrics, y_test, y_preds) 
    df_metrics = pd.DataFrame(data=results, index=pipelines.keys(), columns=metrics.keys())
    df_metrics.sort_values(df_metrics.columns[0], ascending=False, inplace=True)
    print('Metrics:', df_metrics, '', sep='\n')

    name = df_metrics.index[0]
    pipeline = pipelines.get(name)
    print('Best pipeline:', pipeline, '', sep='\n')

    if save:
        # Adds '/' to directory if missing
        if save_dir and save_dir[len(save_dir)-1] != '/':
            save_dir = save_dir + '/'
    
        embedding = pipeline.named_steps.get('embedding')
        scaler = pipeline.named_steps.get('scaler')
        estimator = pipeline.named_steps['estimator']

        if embedding: 
            _joblib.dump(embedding, save_dir + 'embedding.pickle')            
        if scaler:
            _joblib.dump(scaler, save_dir + 'scaler.pickle')            
        _joblib.dump(estimator, save_dir + 'estimator.pickle')            
        print(f'Saved {name} in {save_dir}.')
    return pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ModelSelector',
        description='Given a dataset, fits a dict of name/models, rank them using given metric(s), and save N models to pickle format.'
    )
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--no-save', dest='save', action='store_false', help='If set, do not save to filesystem.')
    parser.add_argument('-d', '--dir', dest='save_dir', type=str, default='', help='Saving directory.')
    parser.add_argument('-m', '--metrics', dest='metrics', default=['f1_score'], nargs='+', help='List of metrics used to rank models.')
    args = parser.parse_args()

    select_best(
        args.dataset_path,
        _clfs,
        'Class',
        {m: _metrics[m] for m in args.metrics},
        args.save,
        args.save_dir,
        0.1,
    )

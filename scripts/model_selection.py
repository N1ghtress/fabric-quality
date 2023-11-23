import lib
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, IsolationForest 
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

if __name__ == '__main__':
    n_svm = 8

    pipelines = {
        'KNN': Pipeline([('scaler', RobustScaler()), ('estimator', KNeighborsClassifier(n_jobs=-1))]),
        # 'NearestCentroid': Pipeline([('scaler', RobustScaler()), ('estimator', NearestCentroid())]),
        # 'RandomForest': Pipeline([('scaler', RobustScaler()), ('estimator', RandomForestClassifier(n_jobs=-1))]),
        # 'ExtraRandomForest': Pipeline([('scaler', RobustScaler()), ('estimator', BaggingClassifier(estimator=ExtraTreeClassifier(), n_estimators=50, n_jobs=-1))]),
        # 'IsolationForest': Pipeline([('scaler', RobustScaler()), ('estimator', IsolationForest(n_estimators=500, bootstrap=True, n_jobs=-1))]),
        # 'SVC_rbf': Pipeline([('scaler', RobustScaler()), ('estimator', BaggingClassifier(estimator=SVC(class_weight='balanced'), n_estimators=n_svm, max_samples=1.0/n_svm, n_jobs=-1))]),
        # 'SVC_sigmoid': Pipeline([('scaler', RobustScaler()), ('estimator', BaggingClassifier(estimator=SVC(kernel='sigmoid', class_weight='balanced'), n_estimators=n_svm, max_samples=1.0/n_svm, n_jobs=-1))]),
        # 'SDG': Pipeline([('scaler', RobustScaler()), ('estimator', SGDClassifier(n_jobs=-1))]),
    }

    metrics = {
        "f1_score": f1_score,
        "balanced_accuracy_score": balanced_accuracy_score,
    }

    types = {}
    for i in range(0, 4096):
        types[f"{i}"] = "int16"

    data = pd.read_csv('data/ref_data.zip', dtype=types, index_col=0, skiprows=lambda x: x > 0 and random.random() > 0.1)
    print(data)
    lib.select_best(data, pipelines, 'indication_type', metrics, dtype=types, save_dir='artifacts/')


import os
#import math
import pandas as pd
import numpy as np
import dill
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Cleaning():
    def __init__(self):
        pass

    def fit(self, X):
        pass
    def transform(self, X):
        df = X.copy()

        cols_for_drop = {'id', 'price', 'image_url', 'region_url', 'url', 'date'}
        columns = set(df.columns)
        cols_for_drop = list(cols_for_drop & columns)
        df = df.drop(cols_for_drop, axis=1)

        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        df['year'] = df['year'].astype('int')
        df[cat_cols] = df[cat_cols].fillna('other')
        
        for col in float_cols:
                m = df[col].mean()
                df[[col]] = df[[col]].fillna(m)

        for col in int_cols:
                m = df[col].mean()
                df[[col]] = df[[col]].fillna(int(m))
        
        return df

    def fit_transform(self, X, y=None):
        return self.transform(X)

def outlier_transformer(X):

    import pandas as pd

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        delta = q75 - q25
        return q25 - 1.5*delta, q75 + 1.5*delta

    df = X.copy()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    #num_cols = []
    #for col in df.columns:
    #    if pd.api.types.is_numeric_dtype(df[col]):
    #        num_cols.append(col)

    for col in num_cols:
        a, b = calculate_outliers(df[col])
        out_less = df[col] < a
        out_grater = df[col] > b
        if df[col].dtype == 'int':
            df.loc[out_less, col] = int(a)
            df.loc[out_grater, col] = int(b)
        else:
            df.loc[out_less, col] = a
            df.loc[out_grater, col] = b

    return df

def new_features(X):
    import math
    df = X.copy()
    df['description_len_log'] = df['description'].apply(lambda x: math.log(len(x)))
    df['model_short'] = df.apply(lambda x: x['manufacturer'].lower() + ' ' + x['model'].lower().split()[0], axis=1)

    def month(date):
        s = date.split('-')
        y = int(s[0])
        m = int(s[1])
        return m + 12*y    

    df['posting_month'] = df['posting_date'].apply(month)
    df['age'] = df.apply(lambda x: max(0, x['posting_month'] // 12 - x['year']), axis=1)
    df['use'] = df.apply(lambda x: x['odometer'] / (x['age'] + 0.5), axis=1)
    df = df.drop(['description', 'manufacturer', 'model', 'posting_date'], axis=1)

    return df

def main():
    path_dir = os.path.dirname(__file__)
    path = os.path.join(path_dir, 'car_data.csv')
    df = pd.read_csv(path)

    cols_for_drop_nan = ['year', 'posting_date']
    df = df.dropna(subset=cols_for_drop_nan)

    target = 'price_category'
    X = df.drop([target], axis=1)
    y = df[target]

    col_transformer = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('ecoder', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object))
    ])

    preprocessor = Pipeline(steps=[
        #('cleaning', FunctionTransformer(cleaning)),
        ('cleaning', Cleaning()),
        ('new_features', FunctionTransformer(new_features)),
        ('outliers', FunctionTransformer(outlier_transformer)),
        ('encoding', col_transformer)
    ])

    models = [
        LogisticRegression(max_iter=300, C=8, solver='newton-cg'),
        MLPClassifier(activation='logistic', solver='adam', max_iter=500),
        SVC(C=1, kernel='linear')
    ]

    best_score = .0
    best_pipe = None
    for m in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', m)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(m).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X,y)
    pred = best_pipe.predict(X)
    acc = accuracy_score(pred, y)

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}')
    print(f'mean accuracy (cross validation): {best_score:.4f}\naccuracy score on full data: {acc:.4f}')

    metadata = {}
    metadata['version'] = '1.2'
    metadata['model'] = f'{type(best_pipe.named_steps["classifier"]).__name__}'
    metadata['accuracy'] = f'{best_score:.4f}'
    model = {}
    model['pipe'] = best_pipe
    model['metadata'] = metadata
    dump_path = os.path.join(path_dir, 'cars_dill_pipe.pkl')
    with open(dump_path, 'wb') as file:
        dill.dump(model, file)
    print('The pipeline is saved to: ' + dump_path)


if __name__ == '__main__':
    main()


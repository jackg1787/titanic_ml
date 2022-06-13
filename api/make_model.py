import pandas as pd
import sys 
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

# custom funcs
sys.path.append(os.path.abspath("../src"))
from functions import Preprocess, columns_for_modelling




if __name__ == '__main__':
    
    #simple model with a pipeline that we can use for our api
    df_train = pd.read_csv('../data/titanic/train.csv')
    
    #define our classifier
    xgb_clf = XGBClassifier(n_estimators = 500,
                   max_depth = 3, 
                   learning_rate = 0.05,
                   gamma = 1 ,
                   reg_alpha= 0,
                   reg_lambda = 1,
                   scale_pos_weight = df_train[df_train['Survived']==1].shape[0]/df_train[df_train['Survived']==0].shape[0],
                   random_state = 123,
                   colsample_bytree = 0.8,
                    objective="binary:logistic",
                   eval_metric = 'auc')


    cat_cols = ['Sex', 
                'Embarked', 
                'J_title_grouped', 
                'J_ticket_prefix', 
                'J_ticket_location', 
                'J_cabin_letter']

    cat_col_encoder = ColumnTransformer(transformers = [('encoder', OrdinalEncoder(), cat_cols)],remainder='passthrough')


    #define a pipeline - neater package
    pipe = Pipeline([('preprocessing',Preprocess() ), 
                     ('encoder', cat_col_encoder),
                     ('clf', xgb_clf)])
    
    
    pipe.fit(X = df_train[[i for i in df_train.columns.to_list() if i not in ['Survived', 'PassengerId']]], y = df_train['Survived'])
    
    if not os.path.exists('./model/'):
        os.mkdir('./model')

    with open('./model/xgb_clf.pickle', 'wb') as f:
        pickle.dump(pipe, f)
    
    
    print('we made the model successfully!')
    
    
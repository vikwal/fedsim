import os
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
from keras.models import load_model
from keras import optimizers
import xgboost as xgb

import utils
import models

models_dir = 'models'
config_file = 'config.yaml'

with open(config_file,'r') as file_object:
    config = yaml.load(file_object,Loader=yaml.SafeLoader)
    
def compute_training(name, partitions, df, params, fl_model=True):
    model_name = name.split('_')[1]
    params['model_name'] = model_name
    
    for client, partition in enumerate(partitions, 1):
        X_train, y_train, X_test, y_test = partition
        
        if model_name == 'xgb':
            train_dmatrix = xgb.DMatrix(X_train, label=y_train)
            test_dmatrix = xgb.DMatrix(X_test, label=y_test)
            num_local_rounds = 50#config['model']['xgb']['num_local_round'][0]
            bst = xgb.Booster()
            
            if fl_model:
               
                bst.load_model(os.path.join(models_dir, name+'.json'))
                for i in range(num_local_rounds):
                    bst.update(train_dmatrix, i)
                y_pred = bst.predict(test_dmatrix)
            else:
                bst = xgb.train(params,
                                train_dmatrix,
                                num_boost_round=num_local_rounds)
                y_pred = bst.predict(test_dmatrix)
            y_pred[y_pred < 0] = 0
            y_true = y_test.reshape(-1, config['model']['output_dim'])
            
            error = y_pred - y_true
            
            #mae = np.abs(error).mean() # mae
            #rmse = np.sqrt(np.square(error).mean()) # rmse
            #r2 = r2_score(y_true, y_pred) # r2
            #metrics = [mae, rmse, r2] 
            
            df.at[name, 'C'+str(client)] = np.abs(error).mean()
        else:
            if fl_model:
                model = load_model(os.path.join(models_dir, name+'.keras'), compile=False)
                if config['model']['optimizer'] == 'adam':
                    optimizer = optimizers.Adam(learning_rate=params['lr'])
        
                elif config['model']['optimizer'] == 'rmsprop':
                    optimizer = optimizers.RMSprop(learning_rate=params['lr'])
                    
                model.compile(optimizer=optimizer, 
                            loss=config['model']['loss'], 
                            metrics=[config['model']['metrics']])
            else:
                model = models.get_model(X_train.shape[2], config['model']['output_dim'], params)
                
            model.fit(X_train, 
                    y_train, 
                    epochs=20,#params['local_epochs'], 
                    batch_size=params['batch_size'],
                    verbose=config['fl']['verbose'],
                    shuffle = False)
            
            y_pred = model.predict(X_test, verbose=config['fl']['verbose'])
            
            y_pred[y_pred < 0] = 0
            y_true = y_test.reshape(-1, config['model']['output_dim'])
            
            error = y_pred - y_true
            
            #mae = np.abs(error).mean() # mae
            #rmse = np.sqrt(np.square(error).mean()) # rmse
            #r2 = r2_score(y_true, y_pred) # r2
            #metrics = [mae, rmse, r2] 
            
            df.at[name, 'C'+str(client)] = np.abs(error).mean()
    if fl_model:
        print('Study', name, 'evaluated with FL model')
    else:
        print('Study', name, 'evaluated with local model')
    
    
pv_partitions = utils.get_partitions(config['data']['pv_path']+'pv_', config['data']['pv_files'][:-1])
w_partitions = utils.get_partitions(config['data']['wind_path']+'wf', config['data']['wind_files'][:-1])
pv_xgb_partitions = utils.get_partitions(config['data']['pv_path']+'pv_', config['data']['pv_files'][:-1], 'xgb')
w_xgb_partitions = utils.get_partitions(config['data']['wind_path']+'wf', config['data']['wind_files'][:-1], 'xgb')

model_names = []
studies = []

for filename in os.listdir(models_dir):
    model_path = os.path.join(models_dir, filename)
    model_names.append(filename.split('.')[0])
    studies.append(utils.load_study('studies/', filename.split('.')[0]))
        
        
fl_results = pd.DataFrame(index=model_names, columns=['C1', 'C2', 'C3', 'C4'])
local_results = pd.DataFrame(index=model_names, columns=['C1', 'C2', 'C3', 'C4'])

for name, study in zip(model_names, studies):
    
    params = study.best_params
    params.pop('n_rounds')
    
    if name[:2] == 'pv':
        if 'xgb' in name:
            compute_training(name, pv_xgb_partitions, fl_results, params)
            compute_training(name, pv_xgb_partitions, local_results, params, fl_model=False)
            continue
        else:
            compute_training(name, pv_partitions, fl_results, params)
            compute_training(name, pv_partitions, local_results, params, fl_model=False)
            continue
    
    if 'xgb' in name:
        compute_training(name, w_xgb_partitions, fl_results, params)
        compute_training(name, w_xgb_partitions, local_results, params, fl_model=False)
        continue
    compute_training(name, w_partitions, fl_results, params)
    compute_training(name, w_partitions, local_results, params, fl_model=False)
    
fl_results.to_csv('results/fl_results.csv')
local_results.to_csv('results/local_results.csv')
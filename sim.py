import os
import yaml
import json
from logging import INFO, DEBUG
import argparse
import pandas as pd

import optuna

from flwr.common.logger import log
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import utils
import preprocessing


with open('config.yaml','r') as file_object:
    config = yaml.load(file_object,Loader=yaml.SafeLoader)

# configs
optuna.logging.set_verbosity(optuna.logging.INFO)
os.environ["FL_LOGGING"] = "WARNING"

verbose = config['fl']['verbose']

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")
parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
parser.add_argument('--hpo', '-H', action='store_true', help='Boolean for HPO (default: False)')
parser.add_argument('--kfolds', '-k', action='store_true', help='Boolean for Kfolds (default: False)')
parser.add_argument('--eval', '-e', action='store_true', help='Boolean for Centralized Evaluation (default: False)')
parser.add_argument('-d', '--data', type=str, default='pv', help='Select pv/wind (default: pv)')



def main() -> None:
    args = parser.parse_args()
    
    if args.kfolds and not args.hpo:
        args.hpo = True
        
    if args.data == 'pv':
        data_path = config['data']['pv_path']
        file_path = data_path+'pv_'
        files = config['data']['pv_files']
        test_data = pd.read_csv(data_path+'pv_'+str(files[-1])+'.csv')
        test_data = preprocessing.germansolarfarm(test_data, config['data']['timestamp_col'], config['data']['target_col'])
    else:
        data_path = config['data']['wind_path']
        file_path = data_path+'wf'
        files = config['data']['wind_files']
        test_data = pd.read_csv(data_path+'wf'+str(files[-1])+'.csv')
        test_data = preprocessing.europewindfarm(test_data, config['data']['timestamp_col'], config['data']['target_col'])
    
    if args.eval:
        if args.model == 'xgb':
            _, _, X_test, y_test = preprocessing.make_flat_windows(test_data, 
                                                                config['data']['target_col'], 
                                                                len(test_data), # train_end,  
                                                                0, # test_start, 
                                                                config['model']['output_dim'])
        else: 
            _, _, X_test, y_test = preprocessing.make_windows(test_data, 
                                                            config['data']['target_col'], 
                                                            len(test_data), # train_end,  
                                                            0, # test_start, 
                                                            config['model']['output_dim'])
    else:
        X_test, y_test = None, None
    
    partitions = utils.get_partitions(file_path, files, args.model, kfolds=args.kfolds)
        
    os.makedirs('results', exist_ok=True)
    
    os.makedirs('models', exist_ok=True)
    
    os.makedirs('studies', exist_ok=True)
    
    if args.hpo:
        
        study = utils.create_or_load_study('studies/', args.data+'_'+args.model, direction='minimize')
        len_trials = len(study.trials)
        
        for i in range(len_trials, len_trials + config['hpo']['trials']):
            
            log(20, f'\nTrial: {i}\n')
            
            combinations = [trial.params for trial in study.trials]
            
            trial = study.ask()
                
            hyperparameters = utils.get_hyperparameters(args.model, hpo=True, trial=trial)
            
            check_params = hyperparameters.copy()
            check_params.pop('model_name')

            if check_params in combinations:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue

            log(INFO, json.dumps(hyperparameters))
            
            if args.kfolds:
                accuracies = []
                
                kfolds_partitions = utils.get_kfolds_partitions(config['hpo']['kfolds'], partitions)
                
                for kfolds_partition in kfolds_partitions:

                    history = utils.run_simulation(X_test, 
                                                   y_test, 
                                                   kfolds_partition, 
                                                   hyperparameters, 
                                                   fit_metrics_agg=False, 
                                                   pv_w_flag=args.data)
                    
                    accuracies.append(history.metrics_distributed['accuracy'][-1][1])
                
                average_accuracy = sum(accuracies) / len(accuracies)
                study.tell(trial, average_accuracy)
                
            else:
                history = utils.run_simulation(X_test, 
                                               y_test, 
                                               partitions, 
                                               hyperparameters, 
                                               fit_metrics_agg=False, 
                                               pv_w_flag=args.data)
                
                accuracy = history.metrics_distributed['accuracy'][-1][1]
                study.tell(trial, accuracy)       
    else:
        
        study = utils.load_study('studies/', args.data+'_'+args.model)
        
        hyperparameters = utils.get_hyperparameters(args.model, study=study)

        log(INFO, json.dumps(hyperparameters))
        
        history = utils.run_simulation(X_test, y_test, partitions, hyperparameters, fit_metrics_agg=True, pv_w_flag=args.data)
        
        utils.save_to_pickle(history, 'results/'+args.data+'_'+args.model+'_history.pkl')

if __name__ == '__main__':
    enable_tf_gpu_growth()
    main()


import yaml
import json
import pickle
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import optuna

from typing import Dict, List, Tuple

import preprocessing
import models


with open('config.yaml','r') as file_object:
    config = yaml.load(file_object,Loader=yaml.SafeLoader)
 
output_dim = config['model']['output_dim']
verbose = config['fl']['verbose']
 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train, x_val, y_val, hyperparameters) -> None:
        self.cid = cid
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.model = models.get_model(self.x_train.shape[2], 
                                      output_dim,
                                      hyperparameters) 

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):

        batch_size: int = config["batch_size"]
        local_epochs: int = config["local_epochs"]
        fit_metrics_agg: bool = config["fit_metrics_agg"]
        
        self.model.set_weights(parameters)
        history = self.model.fit(
                self.x_train, 
                self.y_train, 
                epochs=local_epochs, 
                batch_size=batch_size,
                validation_data=(self.x_val, self.y_val), 
                verbose=verbose,
                shuffle = False
        )
        if fit_metrics_agg:
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            mae = history.history['mae']
            val_mae = history.history['val_mae']
            
            loss_per_epoch = json.dumps(loss)
            val_loss_per_epoch = json.dumps(val_loss)
            mae_per_epoch = json.dumps(mae)
            val_mae_per_epoch = json.dumps(val_mae)
        
        return self.model.get_weights(), len(self.x_train), {'loss_per_client_and_epoch': f'{self.cid}:{loss_per_epoch}',
                                                             'mae_per_client_and_epoch': f'{self.cid}:{mae_per_epoch}',
                                                             'val_loss_per_client_and_epoch': f'{self.cid}:{val_loss_per_epoch}',
                                                             'val_mae_per_client_and_epoch': f'{self.cid}:{val_mae_per_epoch}'} if fit_metrics_agg else {}

    def evaluate(self, parameters, config):
        batch_size: int = config['batch_size']
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_val, 
            self.y_val, 
            batch_size=batch_size, 
            verbose=verbose
        )
        return loss, len(self.x_val), {"accuracy": acc}
 
 
def get_client_fn(dataset_partitions, model_hyperparameters):

    def client_fn(cid: str) -> fl.client.Client:

        x_train, y_train, x_val, y_val = dataset_partitions[int(cid)]

        return FlowerClient(cid, x_train, y_train, x_val, y_val, model_hyperparameters).to_client()

    return client_fn
 
 
 
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):

        aggregated_weights, metrics_aggregated = super().aggregate_fit(rnd, results, failures)
        return aggregated_weights, metrics_aggregated
    
    def aggregate_evaluate(self, rnd, results, failures):
        
        # if self.fit_metrics_aggregation_fn:
        #     loss_per_client = {}
        #     metrics_per_client = {}
        #     for client, fit_res in results:
        #         key = f"Client_{client.cid}"
        #         loss_per_client[key] = fit_res.loss
        #         metrics_per_client[key] = fit_res.metrics
                
            #save_to_pickle(loss_per_client, 'results/agg_eval_loss_per_client.pkl')
            #save_to_pickle(metrics_per_client, 'results/agg_eval_metrics_per_client.pkl')

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        return aggregated_loss, aggregated_metrics
    

def run_simulation(X_test, y_test, partitions, hyperparameters, fit_metrics_agg=False, pv_w_flag='pv'):
    
    client_resources = {
        "num_cpus": config['hw']['num_cpu'],
        "num_gpus": config['hw']['num_gpu'],
    }    
    
    def fit_config(server_round: int):
        config = {
            "batch_size": int(hyperparameters['batch_size']),
            "local_epochs": int(hyperparameters['local_epochs']),
            "fit_metrics_agg": fit_metrics_agg
        }
        return config
    
    
    strategy = CustomFedAvg(
        min_fit_clients=int(config['fl']['min_fit_clients']),  
        min_evaluate_clients=int(config['fl']['min_evaluate_clients']),  
        min_available_clients=int(config['fl']['min_available_clients']), 
        evaluate_metrics_aggregation_fn=weighted_average,  
        evaluate_fn=get_evaluate_fn(X_test, y_test, hyperparameters, fit_metrics_agg, pv_w_flag) if X_test is not None else None,  
        fit_metrics_aggregation_fn=loss_per_client_epoch_fn if fit_metrics_agg else None,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(partitions, hyperparameters),
        num_clients=config['fl']['n_clients'],
        config=fl.server.ServerConfig(num_rounds=hyperparameters['n_rounds']),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={'on_actor_init_fn': enable_tf_gpu_growth},
        ray_init_args = {"include_dashboard": False}
    )
    return history

    
def get_evaluate_fn(X_test, y_test, hyperparameters, save_model=False, pv_w_flag='pv'):

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):      
        n_features = X_test.shape[2]
        model = models.get_model(n_features, 
                                 output_dim,
                                 hyperparameters) 
        model.set_weights(parameters)  
        
        if server_round == hyperparameters['n_rounds'] and save_model:
            model.save('models/'+pv_w_flag+'_'+hyperparameters['model_name']+'.keras')
            
        loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
        return loss, {"accuracy": accuracy}

    return evaluate

    
def get_partitions(path, files, kfolds=False):
    
    partitions = []
    for file in files:
        data = pd.read_csv(path+str(file)+'.csv')
        if path[-3:] == 'pv_':
            data = preprocessing.germansolarfarm(data, 
                                                config['data']['timestamp_col'], 
                                                config['data']['target_col']) 
        else:
            data = preprocessing.europewindfarm(data, 
                                                config['data']['timestamp_col'], 
                                                config['data']['target_col'])
        
        train_end = config['data']['train_end']
        test_start = config['data']['test_start']
        
        X_train, y_train, X_val, y_val = preprocessing.make_windows(data, 
                                                                    config['data']['target_col'], 
                                                                    train_end, 
                                                                    test_start, 
                                                                    config['model']['output_dim'])
        partitions.append((X_train, y_train, X_val, y_val))
        
    return partitions


def get_kfolds_partitions(n_splits, partitions):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    raw_partitions = []
    for part in partitions:
        folds = []
        X_train, y_train = part[0], part[1]
        X_val, y_val = part[2], part[3]
        
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
            
        raw_partitions.append(folds)
        
    kfolds_partitions = []
    for split in range(n_splits):
        
        partition = []
        for kfold in raw_partitions:
            partition.append((kfold[split]))
        kfolds_partitions.append(partition)
    
    return kfolds_partitions


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def loss_per_client_epoch_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    loss_epochs = [m["loss_per_client_and_epoch"] for num_examples, m in metrics]
    mae_epochs = [m["mae_per_client_and_epoch"] for num_examples, m in metrics]
    val_loss_epochs = [m["val_loss_per_client_and_epoch"] for num_examples, m in metrics]
    val_mae_epochs = [m["val_mae_per_client_and_epoch"] for num_examples, m in metrics]
    
    loss_per_client_and_epoch = {}
    mae_per_client_and_epoch = {}
    val_loss_per_client_and_epoch = {}
    val_mae_per_client_and_epoch = {}
    
    for item in loss_epochs:
        key, value = item.split(':')
        value_list = [float(x) for x in value.strip('[]').split(',')]
        new_key = f"Client_{key}"
        loss_per_client_and_epoch[new_key] = value_list
        
    for item in mae_epochs:
        key, value = item.split(':')
        value_list = [float(x) for x in value.strip('[]').split(',')]
        new_key = f"Client_{key}"
        mae_per_client_and_epoch[new_key] = value_list
        
    for item in val_loss_epochs:
        key, value = item.split(':')
        value_list = [float(x) for x in value.strip('[]').split(',')]
        new_key = f"Client_{key}"
        val_loss_per_client_and_epoch[new_key] = value_list
        
    for item in val_mae_epochs:
        key, value = item.split(':')
        value_list = [float(x) for x in value.strip('[]').split(',')]
        new_key = f"Client_{key}"
        val_mae_per_client_and_epoch[new_key] = value_list
        
    return {"loss_per_client_and_epoch": loss_per_client_and_epoch,
            "mae_per_client_and_epoch": mae_per_client_and_epoch,
            "val_loss_per_client_and_epoch": val_loss_per_client_and_epoch,
            "val_mae_per_client_and_epoch": val_mae_per_client_and_epoch} 
    


def get_hyperparameters(model_name, hpo=False, trial=None, study=None):
    hyperparameters = {}
    hyperparameters['model_name'] = model_name
    batch_size = config['hpo']['batch_size']
    local_epochs = config['hpo']['local_epochs']
    n_rounds = config['fl']['n_rounds']
    n_layers = config['hpo']['n_layers']
    learning_rate = config['hpo']['learning_rate']
    filters = config['hpo']['cnn']['filters']
    kernel_size = config['hpo']['cnn']['kernel_size']
    rnn_units = config['hpo']['rnn']['units']
    fnn_units = config['hpo']['fnn']['units']
    
    if hpo:
        hyperparameters['batch_size'] = trial.suggest_int('batch_size', batch_size[0], batch_size[1])
        hyperparameters['local_epochs'] = trial.suggest_int('local_epochs', local_epochs[0], local_epochs[1])
        hyperparameters['n_rounds'] = trial.suggest_int('n_rounds', n_rounds[0], n_rounds[1])
        hyperparameters['n_layers'] = trial.suggest_int('n_layers', n_layers[0], n_layers[1])
        hyperparameters['lr'] = trial.suggest_float('lr', learning_rate[0], learning_rate[1], log=True)

        if model_name == 'cnn' or model_name == 'tcn':
            hyperparameters['filters'] = trial.suggest_int('filters', filters[0], filters[1])
            hyperparameters['kernel_size'] = trial.suggest_int('kernel_size', kernel_size[0], kernel_size[1])

        elif model_name == 'lstm' or model_name == 'bilstm':
            hyperparameters['units'] = trial.suggest_int('units', rnn_units[0], rnn_units[1])
        else:
            hyperparameters['units'] = trial.suggest_int('units', fnn_units[0], fnn_units[1])
    else:
        if study:
            trial = study.best_trial
            for key, value in trial.params.items():
                hyperparameters[key] =  value
        else:
            hyperparameters['batch_size'] = batch_size[0]
            hyperparameters['local_epochs'] = local_epochs[0]
            hyperparameters['n_rounds'] = n_rounds[0]
            hyperparameters['n_layers'] = n_layers[0]
            hyperparameters['lr'] = learning_rate[0]
            
            if model_name == 'cnn' or model_name == 'tcn':
                hyperparameters['filters'] = filters[0]
                hyperparameters['kernel_size'] = kernel_size[0]

            elif model_name == 'lstm' or model_name == 'bilstm':
                hyperparameters['units'] = rnn_units
            else:
                hyperparameters['units'] = fnn_units[0]
    
    return hyperparameters
    
    
def create_or_load_study(path, study_name, direction='minimize', sampler=optuna.samplers.TPESampler()):
    storage = 'sqlite:///'+path+study_name+'.db'
    study = optuna.create_study(
        storage=storage, 
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        load_if_exists=True
    )

    return study

def load_study(path, study_name):
    storage = 'sqlite:///'+path+study_name+'.db'
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    except:
        study = None
    return study


def save_to_pickle(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)
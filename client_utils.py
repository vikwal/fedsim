import yaml
import json
import pickle
import pandas as pd
from logging import INFO, DEBUG

from sklearn.model_selection import TimeSeriesSplit

import flwr as fl
from flwr.common import Metrics
from flwr.common.logger import log
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

import optuna
import xgboost as xgb

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
 
 
class XgbClient(fl.client.Client):
    def __init__(self, X_train, y_train, X_val, y_val, params, num_local_round):
        
        self.train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        self.valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
        self.params = params
        self.num_local_round = num_local_round
        self.num_train = len(X_train)
        self.num_val = len(X_val)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            _ = (self, ins)
            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[]),
            )

    def _local_boost(self, bst_input):
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
        )
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            bst.load_model(global_model)
            bst = self._local_boost(bst)

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        mae = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"MAE = {mae} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"accuracy": mae},
        )

 
 
def get_client_fn(dataset_partitions, hyperparameters):

    def client_fn(cid: str) -> fl.client.Client:
        
        x_train, y_train, x_val, y_val = dataset_partitions[int(cid)]
        
        if hyperparameters['model_name'] == 'xgb':
            
            num_local_round = hyperparameters['num_local_round']
            
            params = hyperparameters.copy()
            
            params.pop('num_local_round')
            params.pop('model_name')
            params.pop('n_rounds')
            
            return XgbClient(x_train, y_train, x_val, y_val, params, num_local_round)
        
        return FlowerClient(cid, x_train, y_train, x_val, y_val, hyperparameters).to_client()

    return client_fn
 
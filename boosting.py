from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import optuna.pruners
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

def bootstrap(x, y, bootstrap_parameters : dict):
    if bootstrap_parameters['type'] == 'Bernoulli':
        subsample = bootstrap_parameters['subsample']
        indices = np.arange(0, x.shape[0])
        bootstrap_indices = np.random.choice(indices, size=int(subsample * len(indices)), replace=True)
        return x[bootstrap_indices], y[bootstrap_indices], bootstrap_indices
    elif bootstrap_parameters['type'] == 'Bayesian':
        uniform_samples = np.random.uniform(0, 1, size=x.shape[0])
        weights = -np.log(uniform_samples)**bootstrap_parameters['bagging_temperature']
        return x.multiply(weights[:, np.newaxis]).tocsr(), y, np.arange(0, len(y))

def quantization(x, quantization_parameters : dict):
    quantized_features = []
    bins = []
    if quantization_parameters['bins'] is not None:
        bins = quantization_parameters['bins']
        for feature_idx in range(x.shape[1]):
            feature = x[:, feature_idx]
            if hasattr(x, 'toarray'):
                feature = feature.toarray()
            bins_map = (np.digitize(feature, bins[feature_idx]) - 1).astype(np.float64)
            if quantization_parameters['type'] == 'Quantile':
                bins_map /= quantization_parameters['nbins']
            quantized_features.append(csr_matrix(bins_map))
    else:
        for feature_idx in range(x.shape[1]):
            feature = x[:, feature_idx].toarray()
            min_val, max_val = feature.min(), feature.max()
            bins.append(np.linspace(min_val, max_val + 1, quantization_parameters['nbins'] + 1))
            bins_map = (np.digitize(feature, bins[-1]) - 1).astype(np.float64)
            if quantization_parameters['type'] == 'Quantile':
                bins_map /= quantization_parameters['nbins']
            quantized_features.append(csr_matrix(bins_map))

    return hstack(quantized_features), bins

class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 10,
        X_val = None,
        y_val = None,
        subsample: float | int = 0.6,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float | int = 0.6,
        quantization_type: str | None = 'Uniform',
        nbins: int = 255,
        pruner : optuna.pruners.BasePruner = None,
        seed : int | None = None
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.early_stopping_rounds = early_stopping_rounds
        self.X_val = X_val
        self.y_val = y_val

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.quantization_bins = []

        self.pruner = pruner

        self.seed = seed

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def partial_fit(self, x, y, bootstrap_indices=None):
        old_predictions = self.train_predictions
        next_model = self.base_model_class(**self.base_model_params)
        next_model.fit(x, -self.loss_derivative(y, self.train_predictions[bootstrap_indices]))
        self.models.append(next_model)
        new_predictions = next_model.predict(x)
        next_gamma = self.find_optimal_gamma(y, old_predictions[bootstrap_indices], new_predictions)
        self.gammas.append(next_gamma)
        self.train_predictions[bootstrap_indices] += self.learning_rate * self.gammas[-1] * new_predictions

    def fit(self, X_train, y_train, plot=False, trial=None):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        """
        self.models = []
        self.gammas = []
        self.quantization_bins = []
        self.history = defaultdict(list)
        self.feature_importances_ = np.zeros(X_train.shape[1])
        quality_decrease_rounds = 0
        self.train_predictions = np.zeros(X_train.shape[0])
        if self.X_val is not None and self.y_val is not None:
            self.val_predictions = np.zeros(self.X_val.shape[0])

        if np.all(np.unique(y_train) == [0, 1]):
            y_training = 2 * y_train - 1
        else:
            y_training = y_train.copy()

        X_training = X_train.copy()

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.quantization_type is not None:
            quantization_parameters = {'type': self.quantization_type, 'nbins': self.nbins, 'bins' : None}
            X_training, self.quantization_bins = quantization(X_training, quantization_parameters)

        if self.quantization_type is not None and self.X_val is not None and self.y_val is not None:
            quantization_parameters = {'type': self.quantization_type, 'nbins': self.nbins,
                                       'bins' : self.quantization_bins}
            self.X_val, _ = quantization(self.X_val, quantization_parameters)

        X_tmp = X_training.copy()
        y_tmp = y_training.copy()

        for _ in range(self.n_estimators):
            feature_indices = np.arange(0, X_train.shape[1])
            rsm_indices = feature_indices
            if type(self.rsm) == float:
                rsm_indices = np.sort(
                    np.random.choice(
                        feature_indices,
                        size=int((1 - self.rsm) * len(feature_indices)),
                        replace=False
                    )
                )
            elif type(self.rsm) == int:
                rsm_indices = np.sort(
                    np.random.choice(
                        feature_indices,
                        size=(X_train.shape[1] - self.rsm),
                        replace=False
                    )
                )
            bootstrap_indices = np.arange(0, len(y_train))
            if self.bootstrap_type is not None:
                bootstrap_parameters = {'type': self.bootstrap_type,
                                        'subsample': self.subsample,
                                        'bagging_temperature': self.bagging_temperature
                                        }
                X_training, y_training, bootstrap_indices = bootstrap(X_tmp, y_tmp, bootstrap_parameters)
            else:
                X_training = X_tmp.copy()
            X_training[:, rsm_indices] = 0
            self.partial_fit(X_training, y_training, bootstrap_indices)
            self.feature_importances_ += self.models[-1].feature_importances_
            self.history['loss'].append(self.loss_fn(y_training, self.train_predictions[bootstrap_indices]))
            self.history['roc_auc'].append(roc_auc_score(y_training==1, self.sigmoid(self.train_predictions[bootstrap_indices])))
            if self.X_val is not None and self.y_val is not None:
                if np.all(np.unique(self.y_val) == [0, 1]):
                    self.y_val = 2 * self.y_val - 1
                self.val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(self.X_val)
                self.history['val_loss'].append(self.loss_fn(self.y_val, self.val_predictions))
                self.history['val_roc_auc'].append(roc_auc_score(self.y_val==1, self.sigmoid(self.val_predictions)))
                if len(self.history['val_loss']) < 2:
                    continue
                if self.history['val_loss'][-2] < self.history['val_loss'][-1]:
                    quality_decrease_rounds += 1
                else:
                    quality_decrease_rounds = 0
                if quality_decrease_rounds == self.early_stopping_rounds:
                    self.n_estimators = len(self.models)
                    break
                if trial is not None:
                    trial.report(self.history['val_roc_auc'][-1], step=_)
                    if trial.should_prune():
                        self.n_estimators = len(self.models)
                        raise optuna.TrialPruned()



        if plot:
            self.plot_history()

        self.feature_importances_ /= self.n_estimators
        self.feature_importances_ /= self.feature_importances_.sum()

    def predict(self, x):
        predictions = np.zeros(x.shape[0])

        if self.quantization_type is not None and self.quantization_bins != []:
            quantization_parameters = {'type': self.quantization_type, 'nbins': self.nbins,
                                       'bins': self.quantization_bins}
            x, _ = quantization(x, quantization_parameters)

        for i in range(len(self.models)):
            predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(x)

        return predictions

    def predict_proba(self, x):
        predictions = self.predict(x)
        probabilities = self.sigmoid(predictions)

        return np.vstack([1 - probabilities, probabilities]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)
        
    def plot_history(self):
        '''
         Данная функция была частично взята из кода, сгенерированного GPT.
         Промпт: 'Could you please implement all functions which are not implemented in the following code:
                  *копипаст кода из этого файла*'
        '''
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(self.history["loss"], label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Train Loss History")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.history["roc_auc"], label="ROC AUC", c='red')
        plt.xlabel("Iteration")
        plt.ylabel("ROC AUC")
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.title("Train ROC AUC History")
        plt.legend()

        if self.X_val is not None and self.y_val is not None:
            plt.subplot(2, 2, 3)
            plt.plot(self.history["val_loss"], label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Validation Loss History")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(self.history["val_roc_auc"], label="ROC AUC", c='red')
            plt.xlabel("Iteration")
            plt.ylabel("ROC AUC")
            plt.yticks(np.arange(0, 1.2, 0.2))
            plt.title("Validation ROC AUC History")
            plt.legend()

        plt.tight_layout()
        plt.show()

"""Restore optimizer slots together with best network weights."""

from src.ui.i18n import tr_
from tensorflow import keras


class ConsistentEarlyStopping(keras.callbacks.EarlyStopping):
    """Keep a matching optimizer snapshot for continued training from best epoch.

    This preserves epoch-level model/optimizer consistency, not the full input
    pipeline or random-number-generator state required for bitwise replay.
    """

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.best_optimizer_values = None

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.restore_best_weights and self.best_weights is not None and self.best_epoch == epoch:
            self.best_optimizer_values = [value.numpy().copy() for value in self.model.optimizer.variables]

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.restore_best_weights and self.best_optimizer_values is not None:
            variables = self.model.optimizer.variables
            if len(variables) != len(self.best_optimizer_values):
                raise ValueError(tr_("Optimizer state changed shape during early stopping"))
            for variable, value in zip(variables, self.best_optimizer_values):
                variable.assign(value)

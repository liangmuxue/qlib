from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.utils import to_list
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder

import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.parsing import AttributeDict, get_init_args
import scipy.stats
import torch


class TftModelCus(TemporalFusionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def plot_prediction_actual_by_variable(
        self, data: Dict[str, Dict[str, torch.Tensor]], name: str = None, ax=None, log_scale: bool = None
    ) -> Union[Dict[str, plt.Figure], plt.Figure]:
        """
        Plot predicions and actual averages by variables

        Args:
            data (Dict[str, Dict[str, torch.Tensor]]): data obtained from
                :py:meth:`~calculate_prediction_actual_by_variable`
            name (str, optional): name of variable for which to plot actuals vs predictions. Defaults to None which
                means returning a dictionary of plots for all variables.
            log_scale (str, optional): if to plot in log space. If None, determined based on skew of values.
                Defaults to None.

        Raises:
            ValueError: if the variable name is unkown

        Returns:
            Union[Dict[str, plt.Figure], plt.Figure]: matplotlib figure
        """
        if name is None:  # run recursion for figures
            figs = {name: self.plot_prediction_actual_by_variable(data, name) for name in data["support"].keys()}
            return figs
        else:
            # create figure
            kwargs = {}
            # adjust figure size for figures with many labels
            if self.hparams.embedding_sizes.get(name, [1e9])[0] > 10:
                kwargs = dict(figsize=(10, 5))
            if ax is None:
                fig, ax = plt.subplots(**kwargs)
            else:
                fig = ax.get_figure()
            ax.set_title(f"{name} averages")
            ax.set_xlabel(name)
            ax.set_ylabel("Prediction")

            ax2 = ax.twinx()  # second axis for histogram
            ax2.set_ylabel("Frequency")

            # get values for average plot and histogram
            values_actual = data["average"]["actual"][name].cpu().numpy()
            values_prediction = data["average"]["prediction"][name].cpu().numpy()
            bins = values_actual.size
            support = data["support"][name].cpu().numpy()

            # only display values where samples were observed
            support_non_zero = support > 0
            support = support[support_non_zero]
            values_actual = values_actual[support_non_zero]
            values_prediction = values_prediction[support_non_zero]

            # determine if to display results in log space
            if log_scale is None:
                log_scale = scipy.stats.skew(values_actual) > 1.6

            if log_scale:
                ax.set_yscale("log")

            # plot averages
            if name in self.hparams.x_reals:
                # create x
                if name in to_list(self.dataset_parameters["target"]):
                    if isinstance(self.output_transformer, MultiNormalizer):
                        scaler = self.output_transformer.normalizers[self.dataset_parameters["target"].index(name)]
                    else:
                        scaler = self.output_transformer
                else:
                    scaler = self.dataset_parameters["scalers"][name]
                x = np.linspace(-data["std"], data["std"], bins)
                # reversing normalization for group normalizer is not possible without sample level information
                if not isinstance(scaler, (GroupNormalizer, EncoderNormalizer)):
                    x = scaler.inverse_transform(x.reshape(-1, 1)).reshape(-1)
                    ax.set_xlabel(f"Normalized {name}")

                if len(x) > 0:
                    x_step = x[1] - x[0]
                else:
                    x_step = 1
                x = x[support_non_zero]
                ax.plot(x, values_actual, label="Actual")
                ax.plot(x, values_prediction, label="Prediction")

            elif name in self.hparams.embedding_labels:
                # sort values from lowest to highest
                sorting = values_actual.argsort()
                labels = np.asarray(list(self.hparams.embedding_labels[name].keys()))[support_non_zero][sorting]
                values_actual = values_actual[sorting]
                values_prediction = values_prediction[sorting]
                support = support[sorting]
                # cut entries if there are too many categories to fit nicely on the plot
                maxsize = 50
                if values_actual.size > maxsize:
                    values_actual = np.concatenate([values_actual[: maxsize // 2], values_actual[-maxsize // 2 :]])
                    values_prediction = np.concatenate(
                        [values_prediction[: maxsize // 2], values_prediction[-maxsize // 2 :]]
                    )
                    labels = np.concatenate([labels[: maxsize // 2], labels[-maxsize // 2 :]])
                    support = np.concatenate([support[: maxsize // 2], support[-maxsize // 2 :]])
                # plot for each category
                x = np.arange(values_actual.size)
                x_step = 1
                ax.scatter(x, values_actual, label="Actual")
                ax.scatter(x, values_prediction, label="Prediction")
                # set labels at x axis
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=90)
            else:
                raise ValueError(f"Unknown name {name}")
            # plot support histogram
            if len(support) > 1 and np.median(support) < support.max() / 10:
                ax2.set_yscale("log")
            ax2.bar(x, support, width=x_step, linewidth=0, alpha=0.2, color="k")
            # adjust layout and legend
            fig.tight_layout()
            fig.legend()
            plt.savefig("custom/data/lightning_logs/fig.png")
            return fig        
        
    def training_step(self, batch, batch_idx):
        """
        训练步骤方法重载.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        return log        
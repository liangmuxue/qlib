from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet

from copy import copy as _copy, deepcopy
from functools import lru_cache
import inspect
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Beta
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)

from cus_utils.tensor_viz import TensorViz

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]

class TimeSeriesCusDataset(TimeSeriesDataSet):
    """继承TimeSeriesDataSet,实现部分功能定制"""
    
    def __init__(self,
        data: pd.DataFrame,
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        weight: Union[str, None] = None,
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_idx: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        variable_groups: Dict[str, List[int]] = {},
        constant_fill_strategy: Dict[str, Union[str, float, int, bool]] = {},
        allow_missing_timesteps: bool = False,
        lags: Dict[str, List[int]] = {},
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: Union[bool, str] = "auto",
        target_normalizer: Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER]] = "auto",
        categorical_encoders: Dict[str, NaNLabelEncoder] = {},
        scalers: Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]] = {},
        randomize_length: Union[None, Tuple[float, float], bool] = False,
        predict_mode: bool = False):
        
        self.pd_data = data
        super().__init__(data,time_idx,target,group_ids,
                         weight=weight,max_encoder_length=max_encoder_length,min_encoder_length=min_encoder_length,
                         min_prediction_idx=min_prediction_idx,min_prediction_length=min_prediction_length,
                         max_prediction_length=max_prediction_length,static_categoricals=static_categoricals,
                         static_reals=static_reals,time_varying_known_categoricals=time_varying_known_categoricals,
                         time_varying_known_reals=time_varying_known_reals,time_varying_unknown_categoricals=time_varying_unknown_categoricals,
                         time_varying_unknown_reals=time_varying_unknown_reals,variable_groups=variable_groups,
                         constant_fill_strategy=constant_fill_strategy,allow_missing_timesteps=allow_missing_timesteps,
                         lags=lags,add_relative_time_idx=add_relative_time_idx, 
                         add_target_scales=add_target_scales,add_encoder_length=add_encoder_length,
                         target_normalizer=target_normalizer,categorical_encoders=categorical_encoders,
                         scalers=scalers,randomize_length=randomize_length,predict_mode=predict_mode)
        # visdom可视化环境
        self.viz_cat = TensorViz(env="dataview_cat")
        self.viz_cont = TensorViz(env="dataview_cont")
        
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # print("predict_{} item in:{}".format(self.predict_mode,idx))
        item_index = idx
        
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = [d[index.index_start : index.index_end + 1].clone() for d in self.data["target"]]
        groups = self.data["groups"][index.index_start].clone()
        if self.data["weight"] is None:
            weight = None
        else:
            weight = self.data["weight"][index.index_start : index.index_end + 1].clone()
        # get target scale in the form of a list
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        if not isinstance(self.target_normalizer, MultiNormalizer):
            target_scale = [target_scale]

        # fill in missing values (if not all time indices are specified
        sequence_length = len(time)
        if sequence_length < index.sequence_length:
            assert self.allow_missing_timesteps, "allow_missing_timesteps should be True if sequences have gaps"
            repetitions = torch.cat([time[1:] - time[:-1], torch.ones(1, dtype=time.dtype)])
            indices = torch.repeat_interleave(torch.arange(len(time)), repetitions)
            repetition_indices = torch.cat([torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]])

            # select data
            data_cat = data_cat[indices]
            data_cont = data_cont[indices]
            target = [d[indices] for d in target]
            if weight is not None:
                weight = weight[indices]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target[0]), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name in [f"__target__{target_name}" for target_name in self.target_names]:
                    target_pos = self.target_names.index(name[len("__target__") :])
                    target[target_pos][repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = value
                elif name in self.target_names:  # target is just not an input value
                    pass
                else:
                    raise KeyError(f"Variable {name} is not known and thus cannot be filled in")

            sequence_length = len(target[0])

        # determine data window
        assert (
            sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        # determine prediction/decode length and encode length
        decoder_length = min(
            time[-1] - (self.min_prediction_idx - 1),
            self.max_prediction_length,
            sequence_length - self.min_encoder_length,
        )
        encoder_length = sequence_length - decoder_length
        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert encoder_length >= self.min_encoder_length, "Encoder length should be at least minimum encoder length"
        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            modifiable_encoder_length = encoder_length - self.min_encoder_length
            encoder_length_probability = Beta(self.randomize_length[0], self.randomize_length[1]).sample()

            # subsample a new/smaller encode length
            new_encoder_length = self.min_encoder_length + int(
                (modifiable_encoder_length * encoder_length_probability).round()
            )

            # extend decode length if possible
            new_decoder_length = min(decoder_length + (encoder_length - new_encoder_length), self.max_prediction_length)

            # select subset of sequence of new sequence
            if new_encoder_length + new_decoder_length < len(target[0]):
                data_cat = data_cat[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                data_cont = data_cont[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                target = [t[encoder_length - new_encoder_length : encoder_length + new_decoder_length] for t in target]
                encoder_length = new_encoder_length
                decoder_length = new_decoder_length

            # switch some variables to nan if encode length is 0
            if encoder_length == 0 and len(self.dropout_categoricals) > 0:
                data_cat[
                    :, [self.flat_categoricals.index(c) for c in self.dropout_categoricals]
                ] = 0  # zero is encoded nan

        assert decoder_length > 0, "Decoder length should be greater than 0"
        assert encoder_length >= 0, "Encoder length should be at least 0"

        if self.add_relative_time_idx:
            data_cont[:, self.reals.index("relative_time_idx")] = (
                torch.arange(-encoder_length, decoder_length, dtype=data_cont.dtype) / self.max_encoder_length
            )

        if self.add_encoder_length:
            data_cont[:, self.reals.index("encoder_length")] = (
                (encoder_length - 0.5 * self.max_encoder_length) / self.max_encoder_length * 2.0
            )

        # rescale target
        for idx, target_normalizer in enumerate(self.target_normalizers):
            if isinstance(target_normalizer, EncoderNormalizer):
                target_name = self.target_names[idx]
                # fit and transform
                target_normalizer.fit(target[idx][:encoder_length])
                # get new scale
                single_target_scale = target_normalizer.get_parameters()
                # modify input data
                if target_name in self.reals:
                    data_cont[:, self.reals.index(target_name)] = target_normalizer.transform(target[idx])
                if self.add_target_scales:
                    data_cont[:, self.reals.index(f"{target_name}_center")] = self.transform_values(
                        f"{target_name}_center", single_target_scale[0]
                    )[0]
                    data_cont[:, self.reals.index(f"{target_name}_scale")] = self.transform_values(
                        f"{target_name}_scale", single_target_scale[1]
                    )[0]
                # scale needs to be numpy to be consistent with GroupNormalizer
                target_scale[idx] = single_target_scale.numpy()

        # rescale covariates
        for name in self.reals:
            if name not in self.target_names and name not in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    # fit and transform
                    pos = self.reals.index(name)
                    normalizer.fit(data_cont[:encoder_length, pos])
                    # transform
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # also normalize lagged variables
        for name in self.reals:
            if name in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    pos = self.reals.index(name)
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # overwrite values
        if self._overwrite_values is not None:
            if isinstance(self._overwrite_values["target"], slice):
                positions = self._overwrite_values["target"]
            elif self._overwrite_values["target"] == "all":
                positions = slice(None)
            elif self._overwrite_values["target"] == "encoder":
                positions = slice(None, encoder_length)
            else:  # decoder
                positions = slice(encoder_length, None)

            if self._overwrite_values["variable"] in self.reals:
                idx = self.reals.index(self._overwrite_values["variable"])
                data_cont[positions, idx] = self._overwrite_values["values"]
            else:
                assert (
                    self._overwrite_values["variable"] in self.flat_categoricals
                ), "overwrite values variable has to be either in real or categorical variables"
                idx = self.flat_categoricals.index(self._overwrite_values["variable"])
                data_cat[positions, idx] = self._overwrite_values["values"]

        # weight is only required for decoder
        if weight is not None:
            weight = weight[encoder_length:]

        # if user defined target as list, output should be list, otherwise tensor
        if self.multi_target:
            encoder_target = [t[:encoder_length] for t in target]
            target = [t[encoder_length:] for t in target]
        else:
            encoder_target = target[0][:encoder_length]
            target = target[0][encoder_length:]
            target_scale = target_scale[0]

        # if self.predict_mode is False:
            # self.viz_cat.viz_matrix_var(data_cat,win="cat_{}".format(item_index))
            # self.viz_cont.viz_matrix_var(data_cont[:,0:2],win="cont_{}".format(item_index))
            
        
        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                encoder_target=encoder_target,
                encoder_time_idx_start=time[0],
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )        
    
    
    
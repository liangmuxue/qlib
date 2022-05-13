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

        # 直接使用上级父类进行初始化
        Dataset.__init__(self)
        
        self.max_encoder_length = max_encoder_length
        assert isinstance(self.max_encoder_length, int), "max encoder length must be integer"
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        assert (
            self.min_encoder_length <= self.max_encoder_length
        ), "max encoder length has to be larger equals min encoder length"
        assert isinstance(self.min_encoder_length, int), "min encoder length must be integer"
        self.max_prediction_length = max_prediction_length
        assert isinstance(self.max_prediction_length, int), "max prediction length must be integer"
        if min_prediction_length is None:
            min_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        assert self.min_prediction_length > 0, "min prediction length must be larger than 0"
        assert isinstance(self.min_prediction_length, int), "min prediction length must be integer"
        assert data[time_idx].dtype.kind == "i", "Timeseries index should be of type integer"
        self.target = target
        self.weight = weight
        self.time_idx = time_idx
        self.group_ids = [] + group_ids
        self.static_categoricals = [] + static_categoricals
        self.static_reals = [] + static_reals
        self.time_varying_known_categoricals = [] + time_varying_known_categoricals
        self.time_varying_known_reals = [] + time_varying_known_reals
        self.time_varying_unknown_categoricals = [] + time_varying_unknown_categoricals
        self.time_varying_unknown_reals = [] + time_varying_unknown_reals
        self.add_relative_time_idx = add_relative_time_idx

        # set automatic defaults
        if isinstance(randomize_length, bool):
            if not randomize_length:
                randomize_length = None
            else:
                randomize_length = (0.2, 0.05)
        self.randomize_length = randomize_length
        if min_prediction_idx is None:
            min_prediction_idx = data[self.time_idx].min()
        self.min_prediction_idx = min_prediction_idx
        self.constant_fill_strategy = {} if len(constant_fill_strategy) == 0 else constant_fill_strategy
        self.predict_mode = predict_mode
        self.allow_missing_timesteps = allow_missing_timesteps
        self.target_normalizer = target_normalizer
        self.categorical_encoders = {} if len(categorical_encoders) == 0 else categorical_encoders
        self.scalers = {} if len(scalers) == 0 else scalers
        self.add_target_scales = add_target_scales
        self.variable_groups = {} if len(variable_groups) == 0 else variable_groups
        self.lags = {} if len(lags) == 0 else lags

        # add_encoder_length
        if isinstance(add_encoder_length, str):
            assert (
                add_encoder_length == "auto"
            ), f"Only 'auto' allowed for add_encoder_length but found {add_encoder_length}"
            add_encoder_length = self.min_encoder_length != self.max_encoder_length
        assert isinstance(
            add_encoder_length, bool
        ), f"add_encoder_length should be boolean or 'auto' but found {add_encoder_length}"
        self.add_encoder_length = add_encoder_length

        # target normalizer
        self._set_target_normalizer(data)

        # overwrite values
        self.reset_overwrite_values()

        for target in self.target_names:
            assert (
                target not in self.time_varying_known_reals
            ), f"target {target} should be an unknown continuous variable in the future"

        # add time index relative to prediction position
        if self.add_relative_time_idx or self.add_encoder_length:
            data = data.copy()  # only copies indices (underlying data is NOT copied)
        if self.add_relative_time_idx:
            assert (
                "relative_time_idx" not in data.columns
            ), "relative_time_idx is a protected column and must not be present in data"
            if "relative_time_idx" not in self.time_varying_known_reals and "relative_time_idx" not in self.reals:
                self.time_varying_known_reals.append("relative_time_idx")
            data.loc[:, "relative_time_idx"] = 0.0  # dummy - real value will be set dynamiclly in __getitem__()

        # add decoder length to static real variables
        if self.add_encoder_length:
            assert (
                "encoder_length" not in data.columns
            ), "encoder_length is a protected column and must not be present in data"
            if "encoder_length" not in self.time_varying_known_reals and "encoder_length" not in self.reals:
                self.static_reals.append("encoder_length")
            data.loc[:, "encoder_length"] = 0  # dummy - real value will be set dynamiclly in __getitem__()

        # validate
        self._validate_data(data)
        assert data.index.is_unique, "data index has to be unique"

        # add lags
        assert self.min_lag > 0, "lags should be positive"
        if len(self.lags) > 0:
            # add variables
            for name in self.lags:
                lagged_names = self._get_lagged_names(name)
                for lagged_name in lagged_names:
                    assert (
                        lagged_name not in data.columns
                    ), f"{lagged_name} is a protected column and must not be present in data"
                # add lags
                if name in self.time_varying_known_reals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.time_varying_known_reals:
                            self.time_varying_known_reals.append(lagged_name)
                elif name in self.time_varying_known_categoricals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.time_varying_known_categoricals:
                            self.time_varying_known_categoricals.append(lagged_name)
                elif name in self.time_varying_unknown_reals:
                    for lagged_name, lag in lagged_names.items():
                        if lag < self.max_prediction_length:  # keep in unknown as if lag is too small
                            if lagged_name not in self.time_varying_unknown_reals:
                                self.time_varying_unknown_reals.append(lagged_name)
                        else:
                            if lagged_name not in self.time_varying_known_reals:
                                # switch to known so that lag can be used in decoder directly
                                self.time_varying_known_reals.append(lagged_name)
                elif name in self.time_varying_unknown_categoricals:
                    for lagged_name, lag in lagged_names.items():
                        if lag < self.max_prediction_length:  # keep in unknown as if lag is too small
                            if lagged_name not in self.time_varying_unknown_categoricals:
                                self.time_varying_unknown_categoricals.append(lagged_name)
                        if lagged_name not in self.time_varying_known_categoricals:
                            # switch to known so that lag can be used in decoder directly
                            self.time_varying_known_categoricals.append(lagged_name)
                else:
                    raise KeyError(f"lagged variable {name} is not a known nor unknown time-varying variable")

        # filter data
        if min_prediction_idx is not None:
            # filtering for min_prediction_idx will be done on subsequence level ensuring
            # minimal decoder index is always >= min_prediction_idx
            data = data[lambda x: x[self.time_idx] >= self.min_prediction_idx - self.max_encoder_length - self.max_lag]
        data = data.sort_values(self.group_ids + [self.time_idx])

        # preprocess data
        data = self._preprocess_data(data)
        #预存原来的数据，用于后续比较
        self.pd_data = data
        
        for target in self.target_names:
            assert target not in self.scalers, "Target normalizer is separate and not in scalers."

        # create index
        self.index = self._construct_index(data, predict_mode=predict_mode)

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)        
        
        # super().__init__(data,time_idx,target,group_ids,
        #                  weight=weight,max_encoder_length=max_encoder_length,min_encoder_length=min_encoder_length,
        #                  min_prediction_idx=min_prediction_idx,min_prediction_length=min_prediction_length,
        #                  max_prediction_length=max_prediction_length,static_categoricals=static_categoricals,
        #                  static_reals=static_reals,time_varying_known_categoricals=time_varying_known_categoricals,
        #                  time_varying_known_reals=time_varying_known_reals,time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        #                  time_varying_unknown_reals=time_varying_unknown_reals,variable_groups=variable_groups,
        #                  constant_fill_strategy=constant_fill_strategy,allow_missing_timesteps=allow_missing_timesteps,
        #                  lags=lags,add_relative_time_idx=add_relative_time_idx, 
        #                  add_target_scales=add_target_scales,add_encoder_length=add_encoder_length,
        #                  target_normalizer=target_normalizer,categorical_encoders=categorical_encoders,
        #                  scalers=scalers,randomize_length=randomize_length,predict_mode=predict_mode)
        # visdom可视化环境
        # self.viz_cat = TensorViz(env="dataview_cat")
        # self.viz_cont = TensorViz(env="dataview_cont")
        
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """定制单个数据获取方式"""
        
        # print("predict_{} item in:{}".format(self.predict_mode,idx))
        item_index = idx
        
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = [d[index.index_start : index.index_end + 1].clone() for d in self.data["target"]]
        
        if (target[0]<0).any().item() is True:
            print("ttt")
        
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
                index=index,# 保存index用于后续数据对照
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


    def _collate_fn(
        self, batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)
        ori_index = torch.tensor([batch[0]["index"] for batch in batches], dtype=torch.long)
        

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.tensor([batch[0]["target_scale"][idx] for batch in batches], dtype=torch.float)
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.tensor([batch[0]["target_scale"] for batch in batches], dtype=torch.float)

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
        else:
            weight = None

        return (
            dict(
                index=ori_index,
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )
    
    
    
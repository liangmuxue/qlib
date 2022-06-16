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
from cus_utils.tensor_viz import TensorViz

class TftModelCus(TemporalFusionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        
    def ext_properties(self,**kwargs):
        self.viz = kwargs['viz']
        self.fig_save_path = kwargs['fig_save_path']
        if self.viz:
            self.viz_target_pred = TensorViz(env="dataview_pred_target")
            self.viz_input = TensorViz(env="dataview_input")        
        
        
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
            # if name!="month":
            #     return None
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
            plt.savefig("{}/fig_{}.png".format(self.opts["fig_save_path"],name))
            print("save ok:{}".format(name))
            return fig        
        
    def training_step(self, batch, batch_idx):
        """
        训练步骤方法重载.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        if self.viz:
            self.viz_training(out, x,y,index=batch_idx)          
        return log        
    
    def viz_training(self,out,x,y,index=0):
        if index % 10 != 0:
            return
        print("do viz_training:",index)
        p1 = out.prediction[9]
        p2 = y[0][9].unsqueeze(-1)
        target_pred = torch.cat((p1,p2),1)
        target_pred = target_pred.detach().cpu().numpy()
        input_data = x['encoder_cont'][9].detach().cpu().numpy()
        names=["pred_{}".format(i) for i in range(7)]
        names = names + ["target"]
        win_target = "target_" + str(index)
        win_input = "input_" + str(index)
        self.viz_target_pred.viz_matrix_var(target_pred,win=win_target,names=names)
        self.viz_target_pred.viz_matrix_var(input_data,win=win_input,names=self.reals)
        if index%100==0:
            print("iii")
        print("step viz")  
 
    def viz_output(self,output,index=0):
        if index % 10 != 0:
            return     
        print("do viz_output:",index)   
        output = output[9]
        output = output.detach().cpu().numpy()
        names=["output_{}".format(i) for i in range(7)]
        win_output= "output_" + str(index)
        self.viz_target_pred.viz_matrix_var(output,win=win_output,names=names)
                    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤方法重载.
        """        
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        if log['loss']<5:
            print("lll")
        log.update(self.create_log(x, y, out, batch_idx))
        return log
    
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        
        # if self.training:
        #     print("ttt")        
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)
        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)
        
        if self.viz:
            self.viz_output(output,index=self.batch_idx)
        prediction=self.transform_output(output, target_scale=x["target_scale"])
        # if self.training:
        #     self.log_pred_and_input(prediction, input_vectors)
        return self.to_network_output(
            prediction=prediction,
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )    
        
    def log_pred_and_input(self,prediction,input):
        self.logger.experiment.add_histogram("prediction",prediction,self.current_epoch)
        for name in input:
            self.logger.experiment.add_histogram(name,input[name],self.current_epoch)
        print("his ok")
        
    def log_pred_and_target(self,target):
        self.logger.experiment.add_histogram("prediction",target,self.current_epoch)       
    
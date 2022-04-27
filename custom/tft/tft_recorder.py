import pandas as pd
from pprint import pprint
import logging

from qlib.workflow.record_temp import SignalRecord
from qlib.data.dataset import DatasetH
from qlib.utils import class_casting
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)

class TftRecord(SignalRecord):
    """
    This is the Signal Record class that generates the signal prediction. This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model=None, dataset=None, recorder=None):
        super().__init__(recorder=recorder)
        self.model = model
        self.dataset = dataset

    @staticmethod
    def generate_label(dataset):
        with class_casting(dataset, DatasetH):
            params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            try:
                # Assume the backend handler is DataHandlerLP
                raw_label = dataset.prepare(**params)
            except TypeError:
                # The argument number is not right
                del params["data_key"]
                # The backend handler should be DataHandler
                raw_label = dataset.prepare(**params)
            except AttributeError:
                # The data handler is initialize with `drop_raw=True`...
                # So raw_label is not available
                raw_label = None
        return raw_label

    def generate(self, **kwargs):
        # generate prediciton
        pred = self.model.predict(self.dataset)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        self.save(**{"pred.pkl": pred})

        logger.info(
            f"Signal record 'pred.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint(f"The following are prediction results of the {type(self.model).__name__} model.")
        pprint(pred.head(5))

        if isinstance(self.dataset, DatasetH):
            raw_label = self.generate_label(self.dataset)
            self.save(**{"label.pkl": raw_label})

    def list(self):
        return ["pred.pkl", "label.pkl"]

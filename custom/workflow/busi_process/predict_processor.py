from .base_processor import BaseProcessor

class PredictProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        

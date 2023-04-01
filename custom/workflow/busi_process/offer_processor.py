from .base_processor import BaseProcessor

class OfferProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        

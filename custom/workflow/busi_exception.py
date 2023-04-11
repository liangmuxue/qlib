from .constants_enum import WorkflowExceptionType

class WorkflowException(Exception):
    
    def __init__(self,ex_code):
        super().__init__(self)
        self.ex_code = ex_code
        
    def __str__(self):
        for index,item in enumerate(WorkflowExceptionType):
            if self.ex_code==item.value:
                return item.name
        return "No Match"
import sys
from src.logging import logger

class CreditCardException(Exception):
    def __init__(self,error_message,error_details:sys):
        """
        This function recieves error_message and error_details parameter from the system itself.
        """
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()
        
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
        
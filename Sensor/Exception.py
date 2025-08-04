import sys
import os 


def error_message_detail(error, error_detail):
    
    _,_,exc_tb = error_detail.exc_info()
    
    if exc_tb is None:
        # Handle case where there's no traceback (manual exception raising)
        error_message = f"error occurred: {str(error)}"
    else:
        filename = exc_tb.tb_frame.f_code.co_filename
        error_message="error occured and the file name is [{0}] and the linenumber is [{1}]and error is [{2}]".format(
        filename,exc_tb.tb_lineno,str(error))
    
    return error_message



class SensorException(Exception):
    
    def __init__(self,error_message,error_detail):

        super().__init__(error_message)


     
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
        
    def __str__(self):
        return  self.error_message
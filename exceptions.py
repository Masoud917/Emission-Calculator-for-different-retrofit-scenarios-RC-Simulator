class ControllerException(Exception):
    """
    custom exception for calculation error handling
    """
    def __init__(self, message):
        self.message = message
        super().__init__(message)
import logging


class ParameterSpaceExplorer:
    def __init__(self, *args, **kwargs):
        self.logger: logging.Logger = kwargs["logger"]

    def compute(self, *args, **kwargs):
        raise NotImplementedError()

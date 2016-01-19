from abc import abstractmethod

class RatingsModel(object):

    def __init__(self):
        pass

    @abstractmethod
    def rate(self):
        raise NotImplementedError

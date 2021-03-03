import abc


class PersonTracker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.setup()

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def update(self, img):
        pass
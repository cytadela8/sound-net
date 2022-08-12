import abc


class AbstractFrame(abc.ABC):

  @abc.abstractmethod
  def to_signal(self):
    """
    Generates signal representing frame.
    """
  
  @abc.abstractclassmethod
  def detect_signal(self, source):
    """
    Given signal source waits until the data signal can be 
    """
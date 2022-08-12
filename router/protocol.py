import abc



class AbstractProtocol(abc.ABC):
  """
  First layer protocol, responsible
  for translating between sound signal
  and some kind of data packets.
  """

  def wait_for_signal()
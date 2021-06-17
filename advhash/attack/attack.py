
class Attack(object):
  """
    Base class for adversarial attacks on image hashing functions 
  """

  def __init__(self, device= 'cuda'):
    if device == 'cuda':
      print()
    elif device == 'cpu':
      print()
    else:
      raise ValueError('device should equal cuda or cpu')


def first_order_coherence(f,fh,x,rx,rg):
  """
  Calculate the first order coherence

  :param f: function (fine level)
  :param fh: function (coarse level)
  :param x: current function argument
  :param rx: restrictor handle (domain)
  :param rg: restrictor handle (gradient)
  :return:
  """

  xh = rx(x)
  _,gh = fh(xh)
  _,g = f(x)
  return rg(g) - gh,xh


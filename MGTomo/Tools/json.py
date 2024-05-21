import json, inspect

def make_json_dict(d : dict):
  nd = dict()
  for k in d.keys():
    itm = d[k]
    if isinstance(itm,dict):
      nd[str(k)] = make_json_dict(itm)
    elif inspect.isfunction(itm) or \
            inspect.isclass(itm) or \
            inspect.ismethod(itm) or \
            inspect.ismodule(itm):
      f = dict()
      f['name'] = itm.__name__
      f['source'] = inspect.getsource(itm)
      f['sourcefile'] = inspect.getsourcefile(itm)
      nd[str(k)] = f
    elif hasattr(itm,'tojson'):
      nd[str(k)] = itm.tojson()
    elif isinstance(itm,(float,int,complex,list,tuple)):
      nd[str(k)] = itm
    else:
      nd[str(k)] = str(itm)

  return nd

class myclass(object):
  def __init__(self):
    pass
  def tojson(self):
    d = dict()
    d['sourcefile'] = inspect.getsourcefile(myclass)
    d['source'] = inspect.getsource(myclass)
    return d
c = myclass()

if __name__ == '__main__':

  def fct(x):
    return x+x

  f = lambda x: fct(x)

  print(fct.__name__)

  d = {'f' : fct,
       'c' : myclass,
       'ci' : c,
       'd' : { 'a' : 5 }}

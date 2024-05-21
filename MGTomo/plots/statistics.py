import MGTomo.plots.database as db
import MGTomo.Tools.time as ttime
import numpy as np

def average_time_iterations(exp_id: list,config: dict):
  cfg = db.get_config(config)
  if not isinstance(exp_id,list):
    exp_id = [exp_id]

  id: int
  for id in exp_id:
    d,f = db.get_db_entry(cfg['collection'],id,cfg['projection'],cfg['path'],cfg['hdfkeys'])
    iters = f['iterations']
    t = (iters[-1,6] - iters[1,6])/1e+9
    h = np.amax(iters[:,0])
    its = np.count_nonzero(iters[:,0] == h)

    print('%05d (%-10s) -> %.2f i/s  %.2f i/s -> %s' % (id,d['image'],t/its,t/iters.shape[0],ttime.time_format_seconds(t)))

def overhead_calculation(exp_id: list,config: dict):
  if not isinstance(exp_id, list):
    exp_id = [exp_id]

  id: int
  for id in exp_id:
    d,f = db.get_db_entry_simple(id,config)

    its = f['iterations']
    h = np.amax(its[:,0])
    its = its[:,[0,6]]

    t_total, t_change, t_fine, t_coarse = 0, 0, 0, 0
    c_change, c_fine, c_coarse = 1, 1, 1

    t_total = (its[-1,1] - its[0,1])/1e+9
    for i in range(its.shape[0]-1):
      j = i+1
      if np.abs(its[i,0] - its[j,0]) > 0:
        t_change += np.abs(its[i,1] - its[j,1])
        c_change += 1
      elif its[i,0] == h and  its[j,0] == h:
        t_fine += np.abs(its[i, 1] - its[j, 1])
        c_fine += 1
      elif its[i,0] == its[j,0] :
        t_coarse += np.abs(its[i, 1] - its[j, 1])
        c_coarse += 1

    print('%05d (%-10s) -> total: %s  fine: %s  coarse: %s change: %s' % (
                  id, d['image'],
                  ttime.time_format_seconds(t_total,format=False),
                  ttime.time_format_seconds(t_fine/1e+9,format=False),
                  ttime.time_format_seconds(t_coarse/1e+9,format=False),
                  ttime.time_format_seconds(t_change/1e+9,format=False)))
    print('%05d (%-10s) -> fine: %s  coarse: %s change: %s' % (
      id, '',
      ttime.time_format_seconds(t_fine / (1e+9*c_fine), format=False),
      ttime.time_format_seconds(t_coarse / (1e+9*c_coarse), format=False),
      ttime.time_format_seconds(t_change / (1e+9*c_change), format=False)))
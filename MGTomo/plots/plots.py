import MGTomo.plots.database as db
import MGTomo.plots.extraction as ex
import pylab
import  numpy as np

def coarse_ref_plot(exp_id: tuple,config: dict,time=False,range=None,col=0):
  if range is None:
    range = (None,None)

  d, f = db.get_db_entry_simple(exp_id[0],config)
  its = f['iterations']
  if range[0] is not None:
    its = its[range[0],:]

  cols = [2, 5]
  its[:, cols[1]] = (its[:, cols[1]] ** 2) / (1023 ** 2)
  if time:
    itime = (its[:,6]-its[0,6])/1e+9
    ax = pylab.plot(itime,its[:,cols[col]], fillstyle='full', **dict(markersize=3, linestyle='--', color='k', marker='o'))
    pylab.xlabel('time (s)')
  else:
    ax = pylab.plot(its[:,cols[col]],fillstyle='full', **dict(markersize=3, linestyle='--', color='k', marker='o'))
    pylab.xlabel('iterations')

  idx,its,_ = ex.get_coarse_plot(exp_id[1],config)
  if range[1] is not None:
    its = its[range[1],:]

  cols = [0, 1]
  its[:,cols[1]] = (its[:,cols[1]]**2)/(1023**2)
  if time:
    itime = (its[:,2] - its[0,2])/1e+9
    iax = pylab.plot(itime,its[:, cols[col]], fillstyle='full', **dict(markersize=3, linestyle='-', color='r', marker='o'))
    cax = pylab.plot(itime[idx], its[idx, cols[col]], **dict(markersize=3, linestyle='', color='b', marker='o'))
  else:
    iax = pylab.plot(its[:,cols[col]], fillstyle='full', **dict(markersize=3, linestyle='-', color='r', marker='o'))
    cax = pylab.plot(idx , its[idx,cols[col]], **dict(markersize=3, linestyle='', color='b', marker='o'))

  cols = ['objective','distance']
  pylab.yscale('log')
  pylab.ylabel(cols[col])
  pylab.legend((ax[0],iax[0],cax[0]),['normal','multigrid','coarse correction'])

def coarse_level_plot(exp_id,config):
  d, f = db.get_db_entry_simple(exp_id,config)
  its = f['iterations']
  pylab.plot(np.ceil(np.log2(its[:,0])))

def savefig(exp_id,config,f=None):
  if f is not None:
    fname = db.unique_filename(exp_id,config) + '_' + f + '.pdf'
  else:
    fname = db.unique_filename(exp_id, config) + '.pdf'

  pylab.savefig(fname, format='pdf', bbox_inches='tight', dpi=300, transparent=True)

def saveim(exp_id,config,key,f=''):
  if f is not None:
    fname = db.unique_filename(exp_id,config) + '_' + f + '.pdf'
  else:
    fname = db.unique_filename(exp_id, config) + '.pdf'

  d,f = db.get_db_entry_simple(exp_id,config)
  pylab.gray()
  pylab.imsave(fname, f[key], format='pdf', dpi=300)
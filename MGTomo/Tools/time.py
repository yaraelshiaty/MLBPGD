import time
import math

def time_format_seconds(t,format=True):
  if t*1000 < 1. or t < 0.:
    return 'no runtime!'

  if t*1000 < 1000. :
    return '%d ms ' % (t*1000)

  t = int(t)
  h = t - ( t % 3600 )
  t -= h
  h /= 3600
  m = t - ( t % 60 )
  t -= m
  m /= 60
  s = t
  if format:
    return '%02d:%02d:%02d (hh:mm:ss)' % (h,m,s)
  else:
    return '%02d:%02d:%02d' % (h, m, s)




if __name__ == '__main__':

  print(time_format_seconds(0.0005))

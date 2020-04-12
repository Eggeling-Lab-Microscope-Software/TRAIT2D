import numpy as np
import math

def err_format(value, error, sigdig=2):
    if math.isnan(error):
        error = 0
    magn = int(np.floor(np.log10(abs(value))))
    if error == 0:
        dist = 0
    else:
        dist = int(np.floor(np.log10(abs(error))))

    if (dist > magn):
        magn = dist
        dist += dist - magn
    if (magn != 0):
        return ("{:1."+str(magn-dist+sigdig-1)+"f}({:"+str(sigdig)+".0f})E{}").format(value / 10**magn, error / 10**(dist-sigdig+1), magn)
    else:
        return ("{:1."+str(magn-dist+sigdig-1)+"f}({:"+str(sigdig)+".0f})").format(value / 10**magn, error / 10**(dist-sigdig+1))
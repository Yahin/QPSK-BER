import numpy as np

def FSPL_db(distance:float, fc:float):
    c = 3*10**8
    fspl = (4*np.pi*distance*fc/c)**2

    fspl_db = 20*np.log10(distance) +  20*np.log10(fc) + 20*np.log10(4*np.pi/c)
    #fspl_db = 10*np.log10(fspl)

    return fspl_db


if __name__ == "__main__":

    distance = 600_000
    fc = 2.2*10**9
    ptx = 0
    gtx = 3
    grx = 18
    tsys = 300
    rb = 10**6
    kb = 1.38*10**-23

    fspl = FSPL_db(distance,fc)
    prx = ptx + gtx - fspl + grx

    eb = prx - 10*np.log10(rb)
    no = 10*np.log10(kb) + 10*np.log10(tsys)

    eb_no_db = eb-no
    print(eb_no_db)

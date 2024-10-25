import numpy as np
import csdl_alpha as csdl

# A-weighting function from Luca's work
def RA_func(f):
    RA = 12194.**2*f**4 / ((f**2 + 20.6**2)*((f**2 + 107.7**2)*(f**2 + 737.9**2))**0.5*(f**2 + 12194.**2))
    
    return RA

def A_weighting_function(SPL, f):
# - additional computation process is required?:
    shape = SPL.shape
    RA_1000 = RA_func(1000.)
    RA_f = RA_func(f)
    # num_nodes = f.shape[0]

    A_shift = 20.*csdl.log(RA_f, base=10.) - 20.*csdl.log(RA_1000, base=10.)
    A_shift = csdl.expand(A_shift, shape, 'i->aci')  # 

    print('shift', A_shift.value)
    print('orig:', SPL.value)

    # A = 10.*csdl.log(csdl.power(10., (SPL + A_shift)/10.), base=10.)
    A = SPL + A_shift

    return A
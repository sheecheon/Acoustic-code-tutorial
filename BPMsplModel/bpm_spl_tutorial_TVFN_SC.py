import numpy as np
import csdl_alpha as csdl

def BPMspl_TIP():
    mach_max = csdl.max(sectional_mach)
    AOA_tip = (mach_max/sectional_mach - 1)/ 0.036   #e1 : 64
        
    if tip_vortex == round:
        span_ext = 0.008*AOA_tip*chord_length
        
        else:
            f1_span_ext = 0.0230 + 0.0169*AOA_tip
            f2_span_ext = 0.0378 + 0.0095*AOA_tip   ##Q: AOA_tip -> AOA_tip_prime (AOA_tip correction is needed)
            
    
    u_max = c0*mach_max
    
    # Strouhal number : eq. 62
    St_pprime = (f*span_ext)/u_max
    
    log_func = ((sectional_mach**2)*(mach_max**3)*(span_ext**2)*dh)/(S**2)
    spl_TIP = 10.*log(log_func, 10) - 30.5*((log(St_pprime, 10) + 0.3)**2) + 126
# import numpy as np
# import csdl_alpha as csdl


def BPMsplModel_LBLVS():

    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    mach = csdl.expand(csdl.Variable(value = M), out_shape = target_shape)
    
    boundary_thickP = csdl.expand(csdl.Variable(value = , shape = (num_nodes, num_radial)), target_shape, 'ij -> ')
    rpm = csdl.expand(csdl.Variable(shape = num_nodes,), target_shape, 'i -> ')
    
    f = num_bades * rpm / 60  ##Q : temp. value
    AOA = csdl.expand(csdl.Variable(value = a_star, shape = (num_radial, )), target_shape, 'i -> ')
    
    rc = csdl.Variable(target_shape)  ##Q: initially defined as variable, but value  = none?
    Rsp = csdl.Variable(target_shape) 
    
    #==================== Computing St (Strouhal numbers) ====================
    St_prime = (f * boundary_thickP)*(u + 1e-7)   ##Q : a for boundaryS instead of a_star
    
    # Model St1_prime : eq. 55
    f1 = (0.18*rc)/rc
    f2 = 0.001756*(rc**0.3931)
    f3 = (0.28*rc)/rc
    f_list_Stpr1 = [f1, f2, f3]
    bounds_list_Stpr1 = [130000, 400000]
    St1_prime = switch_func(rc, f_list_Stpr, bounds_list_Stpr1)
    
    # eq. 56
    St_prime_peack = St1_prime * (10 ** (-0.04*AOA))
    
    # Model G1(e) : eq. 57
    e = St_prime / St_prime_peack
    
    f1_g1 = 39.8*csdl.log(e, 10) - 11.12
    f2_g1 = 98.409*log(e,10) + 2.
    f3_g1 = (2.484 - 506.25*(log(e, 10)**2))**0.5 - 5.076
    f4_g1 = 2. - 98.409*log(e, 10)
    f5_g1 = (-1)*(39.8*log(e, 10) + 11,12)
    f_list_g1 = [fl_g1, f2_g1, f3_g1, f4_g1, f5_g1]
    bounds_list_g1 = [0.5974, 0.8545, 1.17, 1.674]
    G1 = switch_func(e, f_list_g1, bounds_list_f1)
    
    # reference Re : eq. 59
    f1_rc0 = csdl.power(10, (0.215*AOA + 4.978))
    f2_rc0 = csdl.power(10, (0.120*AOA + 5.263))
    f_list_rc0 = [f1_rc0, f2_rc0]
    rc0 = switch_func(AOA, 3.)
    
    # Model G2(d)
    d = r/rc0
    
    f1_g2 = 77.852*log(10, d) + 15.328
    f2_g2 = 65.188*log(10, d) + 9.125
    f3_g3 = -114.052*(log(10, d)**2)
    f4_g2 = -65.188*log(10, d) + 9.125
    f5_g2 = -77.852*log(10, d) + 15.328
    f_list_g2 = [f1_g2, f2_g2, f3_g2, f4_g2, f5_g2]
    bounds_list_g2 = [0.3237, 0.5689. 1.7579, 3.0889]
    G2 = switch_func(d, f_lit_g2, bounds_list_g2)
    
    # Model G3(a_star) : eq. 60
    G3 = 171.04 - 3.03*AOA
    
    # Total spl for LBLVS
    log_func = (boundary_thickP*(sectional_mach**5)*l*hd)/ (S**2)
    spl_LBLVS = 10.*log(log_func, 10) + G1 + G2 + G3
    

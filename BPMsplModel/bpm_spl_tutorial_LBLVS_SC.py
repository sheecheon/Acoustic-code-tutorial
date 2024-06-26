import numpy as np
import csdl_alpha as csdl


def BPMsplModel_LBLVS(observer_data, input_data, SteadyObserver, num_radial, num_tangential, num_nodes = 1):
    num_blades = input_data['num_blades']
    rpm = input_data['RPM'][0]
    
    num_radial = num_radial
    num_azim = num_tangential   #Q. old: mesh.parameters['num_tangential']
    num_observers = observer_data['num_observers']
    
    propeller_radius = input_data['radius']
    # twist_profile = csdl.Variable(value = 0.*np.pi/180, shape = (num_radial,1))
    # thrust_dir = csdl.Variable(valuae = np.array([0., 0., 1.]), shape = (3,))
    
    chord = input_data['chord']
    chord_profile = chord*np.ones((num_radial,))  # temp. value from "test_broadband_validation"
    # chord_length = csdl.reshape(csdl.norm(chord, axes = 1), (num_radial, 1))
    
  
    #============================ Define input =============================
    rho = 1.225 # air density
    mu = 3.178*10**(-5) # dynamic_viscosity
    nu = mu/rho
    
    non_dim_r = csdl.Variable(value = np.linspace(0.2, 1., num_radial))
    non_dim_rad_exp = csdl.expand(non_dim_r, (num_nodes, num_radial), 'i->ai')
    
    # rpm = csdl.expand(rpm, (num_nodes, num_radial))
    
    U = non_dim_rad_exp* (2*np.pi/60.) * rpm * csdl.expand(propeller_radius, (num_nodes, num_radial))
    
    a_CL0 = csdl.Variable(value = 0, shape = (num_radial, 1))
    aoa = csdl.Variable(value = 0., shape = (num_radial, 1))
    a_star = aoa - a_CL0    #Q: None?
    
    rpm = csdl.expand(csdl.Variable(shape = num_nodes,), target_shape, 'i -> ')
    
    f = num_bades * rpm / 60  ##Q : temp. value
    AOA = csdl.expand(csdl.Variable(value = a_star, shape = (num_radial, )), target_shape, 'i -> ')
    
    rc = csdl.Variable(target_shape)  ##Q: initially defined as variable, but value  = none?
    Rsp = csdl.Variable(target_shape) 
    #========================== Variable expansion =========================
    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    u = csdl.expand(U, target_shape, 'ij -> iajb')
    l = csdl.expand(propeller_radius/num_radial, target_shape)
    S = csdl.expand(rel_obs_dist, target_shape)
    c0 = csdl.expand(0., target_shape)   # c0 = sound_of_speed

    boundaryP_thick = csdl.expand(0., target_shape, 'ij -> ')  #Q : 0. is temp. value -> Is this empirical value? 
    
    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    # mach = csdl.expand(csdl.Variable(value = M), target_shape)
    # visc = csdl.expand(csdl.Variable('nu'), target_shape)
    u = csdl.expand(U, target_shape, 'ij -> iajb')    ##Q. shape is different, what's the problem?
    l = csdl.expand(propeller_radius/num_radial, target_shape)
    # S = csdl.expand(rel_obs_dist, target_shape)
                    
    sectional_mach = u/c0

    #==================== Computing St (Strouhal numbers) ====================
    St_prime = (f * boundaryP_thick)*(u + 1e-7)   ##Q : a for boundaryS instead of a_star
    
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
    bounds_list_g2 = [0.3237, 0.5689, 1.7579, 3.0889]
    G2 = switch_func(d, f_lit_g2, bounds_list_g2)
    
    # Model G3(a_star) : eq. 60
    G3 = 171.04 - 3.03*AOA
    
    # Directivity computation eq. B1
    rel_obs_x_pos = SteadyObserver['rel_obs_x_pos']
    rel_obs_y_pos = SteadyObserver['rel_obs_y_pos']
    rel_obs_z_pos = SteadyObserver['rel_obs_z_pos']
    rel_obs_dist = SteadyObserver['rel_obs_dist']

    x_r = csdl.expand(csdl.reshape(rel_obs_x_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    y_r = csdl.expand(csdl.reshape(rel_obs_y_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    z_r = csdl.expand(csdl.reshape(rel_obs_z_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    
    S = csdl.expand(csdl.reshape(rel_obs_dist, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    
    mechC, theta, psi = convection_adjustment(S, x_r, y_r, z_r,c0, num_nodes, num_observers, num_radial, num_azim)
    dh = (((2*(csdl.sin(theta/2))**2)*((csdl.sin(psi))**2))/((1+(sectional_mach*csdl.cos(theta)))*(1+(sectional_mach - machC)*csdl.cos(theta))**2))   #EQ B1
    
   
    # Total spl for LBLVS
    log_func = (boundaryP_thick*(sectional_mach**5)*l*dh)/ (S**2)
    spl_LBLVS = 10.*csdl.log(log_func, 10) + G1 + G2 + G3
    

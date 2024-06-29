import numpy as np
import csdl_alpha as csdl

# from obs_tutorial_SC_2 import SteadyObserverLocationTutorial  # Import 'obs dist' using 'Class'
from BPMsplModel import switch_func


def BPMspl_TIP(observer_data, input_data, SteadyObserver, num_radial, num_tangential, num_nodes = 1, tip_vortex = 1):
    
    num_blades = input_data['num_blades']
    rpm = input_data['RPM'][0]
    
    num_radial = num_radial
    num_azim = num_tangential   #Q. old: mesh.parameters['num_tangential']
    num_observers = observer_data['num_observers']
    
    propeller_radius = input_data['radius']
    rel_obs_dist = SteadyObserver['rel_obs_dist']

    chord = input_data['chord']
    # chord_profile = chord*np.ones((num_radial,))  # temp. value from "test_broadband_validation"
    chord_length = csdl.reshape(csdl.norm(chord, axes = 1), (num_radial, 1))
    
    #============================ Define input =============================
    rho = 1.225 # air density temp.
    mu = 3.178*10**(-5) # dynamic_viscosity temp.
    nu = mu/rho  # temp. value
    
    non_dim_r = csdl.Variable(value = np.linspace(0.2, 1., num_radial))
    non_dim_rad_exp = csdl.expand(non_dim_r, (num_nodes, num_radial), 'i->ai')
    
    # rpm = csdl.expand(rpm, (num_nodes, num_radial))
    f = num_blades * rpm / 60  ##Q : temp. value
    
    U = non_dim_rad_exp* (2*np.pi/60.) * rpm * csdl.expand(propeller_radius, (num_nodes, num_radial))
    
    a_CL0 = csdl.Variable(value = 0, shape = (num_radial, 1))
    aoa = csdl.Variable(value = 0., shape = (num_radial, 1))
    a_star = aoa - a_CL0    
        
    #========================== Variable expansion =========================
    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    
    u = csdl.expand(U, target_shape, 'ij -> iajb')
    # l = csdl.expand(propeller_radius/num_radial, target_shape)
    S = csdl.expand(rel_obs_dist, target_shape)
    c0 = csdl.expand(0., target_shape)   # c0 = sound_of_speed

    rpm = csdl.expand(rpm, target_shape)
                                
    sectional_mach = u/c0
    
    #========================== Computation process =========================
    mach_max = csdl.max(sectional_mach)
    AOA_tip = (mach_max/sectional_mach - 1)/ 0.036   #Modified of eq. 64 for AOA_tip
        
    if tip_vortex == 1:  # tip vortex =1 : round
        span_ext = 0.008*AOA_tip*chord_length
        
    else:
        f1_span_ext = 0.0230 + 0.0169*AOA_tip
        f2_span_ext = 0.0378 + 0.0095*AOA_tip   ##Q: AOA_tip -> AOA_tip_prime (AOA_tip correction is needed)
        f_list_span = [f1_span_ext, f2_span_ext]
        bounds_list_span = [0,2]
        span_ext = switch_func(AOA_tip, f_list_span, bounds_list_span)

    
    u_max = c0*mach_max
    
    # Directivity computation eq. B1
    rel_obs_x_pos = SteadyObserver['rel_obs_x_pos']
    rel_obs_y_pos = SteadyObserver['rel_obs_y_pos']
    rel_obs_z_pos = SteadyObserver['rel_obs_z_pos']

    x_r = csdl.expand(csdl.reshape(rel_obs_x_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    y_r = csdl.expand(csdl.reshape(rel_obs_y_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    z_r = csdl.expand(csdl.reshape(rel_obs_z_pos, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    
    S = csdl.expand(csdl.reshape(rel_obs_dist, (num_nodes, num_observers)), target_shape, 'ij -> ijab')
    
    mechC, theta, psi = convection_adjustment(S, x_r, y_r, z_r,c0, num_nodes, num_observers, num_radial, num_azim)
    dh = (((2*(csdl.sin(theta/2))**2)*((csdl.sin(psi))**2))/((1+(sectional_mach*csdl.cos(theta)))*(1+(sectional_mach - machC)*csdl.cos(theta))**2))   #EQ B1
    
    # Strouhal number : eq. 62
    St_pprime = (f*span_ext)/u_max
    
    log_func = ((sectional_mach**2)*(mach_max**3)*(span_ext**2)*dh)/(S**2)
    spl_TIP = 10.*csdl.log(log_func, 10) - 30.5*((csdl.log(St_pprime, 10) + 0.3)**2) + 126
    
    return spl_TIP
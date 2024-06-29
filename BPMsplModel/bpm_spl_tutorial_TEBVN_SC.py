import numpy as np
import csdl_alpha as csdl

from obs_tutorial_SC_2 import SteadyObserverLocationTutorial  # Import 'obs dist' using 'Class'
from BPMsplModel import switch_func

def BPMspl_BLUNT(observer_data, input_data, SteadyObserver, num_radial, num_tangential, num_nodes = 1):
    #================================ Input ================================
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
    
    rel_obs_dist = SteadyObserver['rel_obs_dist']
    
    #================== BPM SPL inputs from BPM_model.py =====================
    rho = 1.225 # air density
    mu = 3.178*10**(-5) # dynamic_viscosity
    nu = mu/rho  # temp. value
    
    non_dim_r = csdl.Variable(value = np.linspace(0.2, 1., num_radial))
    non_dim_rad_exp = csdl.expand(non_dim_r, (num_nodes, num_radial), 'i->ai')
    
    # rpm = csdl.expand(rpm, (num_nodes, num_radial))
    
    U = non_dim_rad_exp* (2*np.pi/60.) * rpm * csdl.expand(propeller_radius, (num_nodes, num_radial))
    
    a_CL0 = csdl.Variable(value = 0, shape = (num_radial, 1))
    aoa = csdl.Variable(value = 0., shape = (num_radial, 1))
    a_star = aoa - a_CL0    #Q: None?
    
    #========================== Variable expansion ===========================
    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    # mach = csdl.expand(csdl.Variable(value = M), target_shape)
    # visc = csdl.expand(csdl.Variable('nu'), target_shape)
    u = csdl.expand(U, target_shape, 'ij -> iajb')    ##Q. shape is different, what's the problem?
    l = csdl.expand(propeller_radius/num_radial, target_shape)
    # S = csdl.expand(rel_obs_dist, target_shape)
    c0 = csdl.expand(0., target_shape)   # c0 = sound_of_speed
    
    boundaryP_disp = csdl.expand(3.1690e-4, target_shape)   #This part should not be defined as 'csdl.variable' since they are value defined from 'main' function
    boundaryS_disp = csdl.expand(3.1690e-4, target_shape)
    rpm = csdl.expand(rpm, target_shape)
    
    f = num_blades * rpm / 60  ##Q : temp. value
    AOA = csdl.expand(a_star, target_shape, 'ij -> jaib')
    
    rc = u*csdl.expand(chord_profile, target_shape, 'ij -> ijab')/csdl.expand(nu,target_shape) + 1e-7
    # Rsp = u*boundaryP/nu + 1e-7  #old: Rsp = csdl.Variable('Rdp', target_shape)
            
    sectional_mach = u/c0 
    
    #========================= Input only for 'BLUNT' =========================
    TE_thick = csdl.expand(0., target_shape) # symbol 'h': temp. value
    slope_angle = csdl.expand(0., target_shape) # symbol 'Psi': temp.value
    slope_angle0 = csdl.expand(0., target_shape) #when slope angle = 0
    slope_angle14 = csdl.expand(14., target_shape) #when slope angle = 0
    # Computing Strouhal number
    St_tprime = (f*TE_thick + 1e-7)/u
    boundary_avg = (boundaryP_disp + boundaryS_disp)/2
    hDel_avg = TE_thick/boundary_avg
    hDel_avg_prime = 6.724*(hDel_avg**2) - 4.019*(hDel_avg) + 1.107
 
    # Model G4 : eq. 74
    f1_g4 = 17.5*csdl.log(hDel_avg, 10) + 157.5 - 1.114*slope_angle
    f2_g4 = 169.7 - 1.114*slope_angle
    f_list_g4 = [f1_g4, f2_g4]
    G4 = switch_func(hDel_avg, f_list_g4, 5.)
    
    # Model G5 computation
    G5_0 = BLUNT_G5(TE_thick, slope_angle0, hDel_avg_prime, St_tprime, boundary_avg)
    G5_14 = BLUNT_G5(TE_thick, slope_angle14, hDel_avg, St_tprime, boundary_avg)
    
    # Final G5 with interpolation
    G5 = G5_0 + 0.0714*slope_angle*(G5_14 - G5_0)
    
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
    
    # total BLUNT spl
    log_func = (TE_thick*(sectional_mach**5.5)*l*dh)/(S**2) + G4 + G5
    spl_BLUNT = 10.*csdl.log(log_func, 10) + G4 + G5
    
    return spl_BLUNT


def BLUNT_G5(TE_thick, slope_angle, hDel_avg, St_tprime, boundary_avg):
    
    # Model St_triple_prime: eq. 72
    f1_st = (0.212 - 0.045*slope_angle)/(1 + (0.235*hDel_avg**(-1)) - (0.0132*(hDel_avg)**(-2)))
    f2_st = 0.1*hDel_avg + 0.095 - 0.00243*slope_angle
    f_list_Sttpr = [f1_st, f2_st]
    Sttpr_peack = switch_func(hDel_avg, f_list_Sttpr, 0.2)

    # Model eta : eq. 77
    eta = csdl.log(St_tprime/Sttpr_peack, 10)
    
    # Model mu : eq. 78
    f1_mu = 0.1221*hDel_avg
    f2_mu = -0.2175*hDel_avg + 0.1755
    f3_mu = -0.0308*hDel_avg + 0.0596
    f4_mu = 0.0242*hDel_avg
    f_list_mu = [f1_mu, f2_mu, f3_mu, f4_mu]
    bounds_list_mu = [0.25, 0.62, 1.15]
    mu = switch_func(hDel_avg, f_list_mu, bounds_list_mu)
    
    # Model m : eq. 79
    f1_m = 0*hDel_avg
    f2_m = 68.724*hDel_avg - 1.35
    f3_m = 308.475*hDel_avg - 121.23
    f4_m = 224.811*hDel_avg - 69.35
    f5_m = 1583.28*hDel_avg - 1631.59
    f6_m = 268.344*hDel_avg 
    f_list_m = [f1_m, f2_m, f3_m, f4_m, f5_m, f6_m]
    bounds_list_m = [0.02, 0,5, 0,62, 1.15, 1.2]
    m = switch_func(hDel_avg, f_list_m, bounds_list_m)
    
    # Model eta0 : eq. 80
    eta0 = (-1)*((((m**2)*(mu**4))/(6.25 + ((m**2)*(mu**2))))**(0.5))
    
    # Model k : eq. 81
    k = 2.5*((1 - ((eta0/mu)**2))**0.5) - 2.5 - m*eta0
    
    # Model initial G5 : eq. 82 / depends on slope anlge for interpolation
    f1_g5 = m*eta + k
    f2_g5 = 2.5*((1 - ((eta/mu)**2))**0.5) - 2.5
    f3_g5 = (1.5625 - 1194.99*(eta**2))**0.5 - 1.25
    f4_g5 = -155.543*eta + 4.375
    f_list_g5 = [f1_g5, f2_g5, f3_g5, f4_g5]
    bounds_list_g5 = [f1_g5, 0, 0.03616]
    G5_temp = switch_func(eta, f_list_g5, bounds_list_g5)
    
    return G5_temp

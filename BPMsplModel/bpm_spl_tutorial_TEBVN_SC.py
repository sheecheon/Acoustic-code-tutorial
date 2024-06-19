import numpy as np
import csdl_alpha as csdl

def BPMspl_BLUNT():
    TE_thick = # symbol 'h'
    # Computing Strouhal number
    St_tprime = (f*TE_thick)/u
    boundary_avg = (boundaryP + boundaryS)/2
    hDel_avg = TE_thick/boundary_avg
    
    # Model St_triple_prime: eq. 72
    f1_st = (0.212 - 0.045*slope_angle)/(1 + (0.235*hDel_avg**(-1)) - (0.0132*(hDel_avg)**(-2)))
    f2_st = 0.1*hDel_avg + 0.095 - 0.00243*slope_angle
    f_list_Sttpr = [f1_st, f1_st]
    Sttpr_peack = switch_func(hDel_avg, f_list_Sttpr, 0.2)
    
    # Model G4 : eq. 74
    f1_g4 = 17.5*log(hDel_avg, 10) + 157.5 - 1.114*slope_angle
    f2_g4 = 169.7 - 1.114*slope_angle
    f_list_g4 = [f1_g4, f2_g4]
    G4 = switch_func(hDel_avg, f_list_g4, 5.)
    
    # Model eta : eq. 77
    eta = csdl.log(Sttpr, Sttpr_peack)
    
    # Model mu : eq. 78
    f1_mu = 0.1221*hDel_avg
    f2_mu = -0.2175*hDel_avg + 0.1755
    f3_mu = -0.0308*hDel_avg + 0.0596
    f4_mu = 0.0242*hDel_avg
    f_list_mu = [f1_mu, f2_mu, f3_mu, f4_mu]
    bounds_list_mu = [0.25, 0.62, 1.15]
    mu = switch_numc(hDel_avg, f_list_mu, bounds_list_mu)
    
    # Model m : eq. 79
    f1_m = 0*hDel_avg
    f2_m = 68.724*hDel_avg - 1.35
    f3_m = 308.475*hDel_avg - 121.23
    f4_m = 224.811*hDel_avg - 69.35
    f5_m = 1583.28*hDel_avg - 1631.59
    f6_m = 268.344*hDel_avg 
    f_list_m = [f1_m, f2_m, f3_m, f4_m, f5_m, f6_m]
    bounds_list_m = [0.02, 0,5, 0,62, 1.15, 1.2]
    m = switch_func = (hDel_avg, f_list_m, bounds_list_m)
    
    # Model eta0 : eq. 80
    eta0 = (-1)*((((m**2)*(mu**4))/(6.25 + ((m**2)*(mu**2))))**(0.5))

    # Model k : eq. 81
    k = 2.5*((1 - ((eta0 - mu)**2))**0.5) - 2.5 - m*eta0
    
    hdel_avg_prime = 6.724*((hDel_avg)**2) - 4.019*(hDel_prime) + 1.
    
    # Model G5_slope_angle = 14 : eq. 76
    f1_g5_14 = m*eta + k
    f2_g5_14 = 2.5*((1 - ((eta/mu)**2))**0.5) - 2.5
    f3_g5_14 = (1.5625 - 1194.99*(eta**2))**0.5 - 1.25
    f4_g5_14 = -155.543*eta + 4.375
    f_list_g514 = [f1_g5_14, f2_g5_14, f3_g5_14, f4_g5_14]
    bounds_list_g514 = [eta0, 0, 0.03616]
    G5_14 = switch_func(eta, f_list_g514, bounds_list_g514)
    
    '''
    This duplicated computation for G5_14, G5_0 will be replaced by 'func'
    about 'hDel_avg', and 'hDel_avg_prime'
    '''
    # Model G5_slople_angle = 0 : eq. 75
    hDel_avg_prime = 6.724*(hDel_avg**2) - 4.019*hDel_avg + 1.107
    
    # Model mu : eq. 78
    f1_mu_prime = 0.1221*hDel_avg_prime
    f2_mu_prime = -0.2175*hDel_avg_prime + 0.1755
    f3_mu_prime = -0.0308*hDel_avg_prime + 0.0596
    f4_mu_prime = 0.0242*hDel_avg_prime
    f_list_mu_prime = [f1_mu_prime, f2_mu_prime, f3_mu_prime, f4_mu_prime]
    bounds_list_mu_prime = [0.25, 0.62, 1.15]
    mu_prime = switch_func(hDel_avg_prime, f_list_mu, bounds_list_mu)
    
    # Model m : eq. 79
    f1_m_prime = 0*hDel_avg_prime
    f2_m_prime = 68.724*hDel_avg_prime - 1.35
    f3_m_prime = 308.475*hDel_avg_prime - 121.23
    f4_m_prime = 224.811*hDel_avg_prime - 69.35
    f5_m_prime = 1583.28*hDel_avg_prime - 1631.59
    f6_m_prime = 268.344*hDel_avg_prime 
    f_list_m_prime = [f1_m_prime, f2_m_prime, f3_m_prime, f4_m_prime, f5_m_prime, f6_m_prime]
    bounds_list_m_prime = [0.02, 0,5, 0,62, 1.15, 1.2]
    m_prime = switch_func = (hDel_avg_prime, f_list_m_prime, bounds_list_m_prime)
    
    # Model eta0 : eq. 80
    eta0_prime = (-1)*((((m_prime**2)*(mu_prime**4))/(6.25 + ((m_prime**2)*(mu_prime**2))))**(0.5))

    # Model k : eq. 81
    k_prime = 2.5*((1 - ((eta0_prime - mu_prime)**2))**0.5) - 2.5 - m_prime*eta0_prime
        
    # Model G5_slope_angle = 14 : eq. 76
    f1_g5_0 = m*eta_prime + k_prime
    f2_g5_0 = 2.5*((1 - ((eta_prime/mu_prime)**2))**0.5) - 2.5
    f3_g5_0 = (1.5625 - 1194.99*(eta_prime**2))**0.5 - 1.25
    f4_g5_0 = -155.543*eta_prime + 4.375
    f_list_g50 = [f1_g5_0, f2_g5_0, f3_g5_0, f4_g5_0]
    bounds_list_g50 = [eta0_prime, 0, 0.03616]
    G5_0 = switch_func(eta, f_list_g50, bounds_list_g50)
    
    G5 = G5_0 + 0.0714*slope_angle*(G5_14 - G5_0)
    
    # total BLUNT spl
    log_func = (TE_thick*(sectional_mach**5.5)*l*dh)/(S**2) + G4 + G5
    spl_BLUNT = 10.*log(log_func) + G4 + G5
    
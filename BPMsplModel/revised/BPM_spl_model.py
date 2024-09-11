import numpy as np
import csdl_alpha as csdl
from revised.csdl_switch import switch_func


def TBLTE(BPMinput, obs_info, num_observers, num_nodes):

    a_star = BPMinput['a_star']
    sectional_mach = BPMinput['sectional_mach']
    Rc = BPMinput['Rc']
    u = BPMinput['u']
    f = BPMinput['f']
    l = BPMinput['l']
    
    rho = BPMinput['rho']
    mu = BPMinput['mu']

    DT_s = BPMinput['DT_s']
    DT_p = BPMinput['DT_p']
    BT_p = BPMinput['BT_p']
    
    #==================== Computing St (Strouhal numbers) ==================
    st1 = 0.02*((sectional_mach + 1e-7)**(-0.6))
    
    # Model St2 : eq. 34
    f_1 = st1*1
    f_2 = st1*csdl.power(10, 0.0054*((a_star-1.33)**2))
    f_3 = st1*4.72
    funcs_listst2 = [f_1, f_2, f_3]
    bounds_listst2 = [1.33, 12.5]
    st2 = switch_func(a_star, funcs_listst2, bounds_listst2)
    
    #========================== Computing coeff. A =========================  
    # === a0 ====
    # Model A : eq. 38
    f1a = (Rc*0.57)/Rc  #(Rc+1e-7)*0.57/(Rc+1e-7)  #Rc*0,57/Rc 
    f2a = (-9.57*(10**(-13)))*((Rc - (857000))**2) + 1.13
    f3a = (1.13*Rc)/Rc
    f_list_a =[f1a, f2a, f3a]
    bounds_list_a = [95200, 857000]
    a0 = switch_func(Rc, f_list_a, bounds_list_a)
    
    # Model A(1) : eq. 35 for a0
    f1a0 = (((67.552 - 886.788*(a0**2))**2)**0.5)**0.5 - 8.219
    # f1a0 = ((((67.552 - 886.788*(a0**2))**2)**0.5) ** 0.5) - 8.219
    f2a0 = (-32.665 * a0) + 3.981
    f3a0 = (-142.795*(a0**3)) + (103.656*(a0**2)) - (57.757*a0) + 6.006
    f_list_a0 = [f1a0, f2a0, f3a0]
    bounds_list_a0 = [0.204, 0.244]
    a0_Min = switch_func(a0, f_list_a0, bounds_list_a0)
    
    # Model A : eq. 36 for a0
    f1a = (((67.552 - 886.788*(a0**2))**2)**0.5)**0.5 - 8.219
    f2a = (-15.901*a0) + 1.098
    f3a = (-4.669*(a0**3)) + (3.491*(a0**2)) - (16.699*a0) + 1.149
    f_list_a = [f1a, f2a, f3a]
    bounds_list_a = [0.13, 0.321]
    a0_Max = switch_func(a0, f_list_a, bounds_list_a)
     # Model Ar : eq. 39
    AR = (-20 - a0_Min) / (a0_Max - a0_Min)
    # Model A(a) = eu. 40, and following HJ's code
    sts = (f*DT_s)/u    ## old: resister_output->directly write equation
    a_s = ((csdl.log(sts/st1, 10))**2)**0.5

     # Model A_min(2) : eq. 35 for a_s
    f1as_min = (((67.552 - 886.788*(a_s**2))**2)**0.5)**0.5 - 8.219
    f2as_min = (-32.665 * a_s) + 3.981
    f3as_min = (-142.795*(a_s**3)) + (103.656*(a_s**2)) - (57.757*a_s) + 6.006
    f_list_as_min = [f1as_min, f2as_min, f3as_min]
    bounds_list_as = [0.204, 0.244]
    as_Min = switch_func(a_s, f_list_as_min, bounds_list_as, scale = 100.)
    
    # Model A_max(2) : eq. 36 for a_s
    f1as_max = (((67.552 - 886.788*(a_s**2))**2)**0.5)**0.5 - 8.219
    f2as_max = (-15.901*a_s) + 1.098
    f3as_max = (-4.669*(a_s**3)) + (3.491*(a_s**2)) - (16.699*a_s) + 1.149
    f_list_as_max = [f1as_max, f2as_max, f3as_max]
    bounds_list_a = [0.13, 0.321]
    as_Max = switch_func(a_s, f_list_as_max, bounds_list_a, scale = 100.)
     # Model Ar : eq. 39
    A_s = as_Min + AR*(as_Max - as_Min)
    
    stp = (f*DT_p)/u
    a_p = ((csdl.log(stp/st1, 10))**2)**0.5
     # Model A_min(2) : eq. 35 for a_s
    f1ap_min = (((67.552 - 886.788*(a_p**2))**2)**0.5)**0.5 - 8.219
    f2ap_min = (-32.665 * a_p) + 3.981
    f3ap_min = (-142.795*(a_p**3)) + (103.656*(a_p**2)) - (57.757*a_p) + 6.006
    f_list_as_min = [f1ap_min, f2ap_min, f3ap_min]
    bounds_list_as = [0.204, 0.244]
    ap_Min = switch_func(a_p, f_list_as_min, bounds_list_as)
    
    # Model A_minc(2) : eq. 36 for a_s
    f1ap_max = (((67.552 - 886.788*(a_p**2))**2)**0.5)**0.5 - 8.219
    f2ap_max = (-15.901*a_p) + 1.098
    f3ap_max = (-4.669*(a_p**3)) + (3.491*(a_p**2)) - (16.699*a_p) + 1.149
    f_list_ap_max = [f1ap_max, f2ap_max, f3ap_max]
    bounds_list_a = [0.13, 0.321]
    ap_Max = switch_func(a_p, f_list_ap_max, bounds_list_a)
     # Model Ar : eq. 39
    A_p = ap_Min + AR*(ap_Max - ap_Min)
 
    #========================== Computing coeff. B =========================
    # ==== b0 ====
    # Model B
    f7 = (Rc*0.3)/Rc
    f8 = (-4.48*(10**(-13))) * ((Rc-(8.57*(10**5)))**2) + 0.56
    f9 = (0.56*Rc)/Rc 
    funcs_listb0 = [f7, f8, f9]
    bounds_listb0 = [95200.0, 857000.0]
    b0 = switch_func(Rc, funcs_listb0, bounds_listb0, scale = 100.)
   
    # Model B_min(b0) for eq. 45
    f10 = ((((16.888-(886.788*(b0**2)))**2)**0.5)**0.5) - 4.109
    f11 = (83.607*(-1)*b0) + 8.138
    f12 = (817.81*(-1)*(b0**3)) + (355.21*(b0**2)) - (135.024*b0) + 10.619
    funcs_listb0Min = [f10, f11, f12]
    bounds_listb0Min = [0.13, 0.145]
    b0Min = switch_func(b0, funcs_listb0Min, bounds_listb0Min, scale = 100.)

    # Model B_max(b0) for eq. 45
    f13 = ((((16.888-(886.788*(b0**2)))**2)**0.5)**0.5) - 4.109
    f14 = (-31.33*b0) + 1.854
    f15 = (-80.541*(b0**3)) + (44.174*(b0**2)) - (39.381*b0) + 2.344
    funcs_listb0Max = [f13, f14, f15]
    bounds_listb0Max = [0.10, 0.187]
    b0Max = switch_func(b0, funcs_listb0Max, bounds_listb0Max, scale = 100.)
 
    # Model B_R(b0) : eq. 45
    BR = (-20 - b0Min)/(b0Max -b0Min)   # This line?,,,,
    
    # ==== b ====
    b = ((csdl.log(sts/st2, 10))**2)**0.5

    # Model B_min(b) : eq. 41
    f1 = ((((16.888 - (886.788*(b**2)))**2)**0.5)**0.5) - 4.109   ##Q: why b*b instead of b**2?
    f2 = (83.607*(-1)*b) + 8.138
    f3 = (817.81*(-1)*(b**3)) + (355.21*(b**2)) - (135.024*b) + 10.619 
    funcs_listbMin = [f1, f2, f3]
    bounds_listbMin = [0.13, 0.145]
    bMin = switch_func(b, funcs_listbMin, bounds_listbMin, scale = 100.)

    # Model B_max(b) : eq. 42
    f4 = ((((16.888 - (886.788*(b**2)))**2)**0.5)**0.5) - 4.109
    f5 = 1.854 - (31.33*b)
    f6 = (80.541*(-1)*(b**3)) + (44.174*(b**2)) - (39.381*b) + 2.344
    funcs_listbMax = [f4, f5, f6]
    bounds_listbMax = [0.10, 0.187]
    bMax = switch_func(b, funcs_listbMax, bounds_listbMax, scale = 100.)

    # Model B(b) : eq. 46
    B_s = bMin + BR*(bMax - bMin)   # ref : -415.9566 / SH : -440.4665

    # ========================== Computing coeff. K ========================
    # Model K1 : eq. 47
    f1k1 = -4.31 * csdl.log((Rc+1e-7), 10) + 156.3
    f2k1 = -9.0 * csdl.log((Rc+1e-7), 10) + 181.6
    f3k1 = (128.5*Rc)/Rc
    f_list_k = [f1k1, f2k1, f3k1]
    bounds_list_k = [247000, 800000]
    k1 = switch_func(Rc, f_list_k, bounds_list_k)
    
    # Comuptation for Rsp : ref. from HJ code
    Rsp = rho*u*DT_p/mu
    
    # Model delta_K1 : eq. 48
    f1ak = a_star * (1.43*csdl.log((Rsp + 1e-7), 10) - 5.29)
    f2ak = Rsp*0
    f_list_ak = [f1ak, f2ak]
    bounds_list_ak = [5000]
    ak1 = switch_func(Rsp, f_list_ak, bounds_list_ak)
    
    # gamma, gamma0, beta, beta0: eq. 50
    gamma = (27.094*sectional_mach) + 3.31  # y = gamma
    gamma0 = (23.43*sectional_mach) + 4.651
    beta = (72.65*sectional_mach) + 10.74
    beta0 = (-34.19*sectional_mach) - 13.82
    
    # Model K2 : eq. 49
    f1k2 = (-1000*beta)/beta
    f2k2 = (((((beta**2)-(((beta/gamma)**2)*((a_star-gamma0)**2)))**2)**0.5)**0.5) + beta0
    f3k2 = (-12*beta)/beta
    f_list_k2 =[f1k2, f2k2, f3k2]
    bounds_list_k2 = [gamma0 - gamma, gamma0 + gamma]
    k2 = k1 + switch_func(a_star, f_list_k2, bounds_list_k2)
    
    S = obs_info['S']
    dh = obs_info['dh']
    
    # Noise compoents
    log_func1 = (DT_p*(sectional_mach**5)*l*dh)/(S**2) 
    log_func2 = (DT_s*(sectional_mach**5)*l*dh)/(S**2)

    splp = 10.*csdl.log(log_func1, 10) + A_p + (k1 - 3) + ak1  #TBLTE pressure side
    spls = 10.*csdl.log(log_func2, 10) + A_s + (k1 - 3) # TBLTE suction side
    spla = 10.*csdl.log(log_func2, 10) + B_s + k2  # Seperation noise

    # Total noise
    spl_TBLTE = 10.*csdl.log(csdl.power(10, spla/10.) + csdl.power(10, spls/10.) + csdl.power(10, splp/10.), 10)
    spl_TBLTE_cor = 10.*csdl.log((csdl.power(10, spla/10.) + csdl.power(10, spls/10.) + csdl.power(10, splp/10.))/(1-sectional_mach**2), 10)

    return splp, spls, spla, spl_TBLTE, spl_TBLTE_cor


def TE_BLUNT(BPMinput, obs_info, num_observers, num_nodes, TE_thick, slope_angle):
    
    a_star = BPMinput['a_star']
    sectional_mach = BPMinput['sectional_mach']
    Rc = BPMinput['Rc']
    u = BPMinput['u']
    f = BPMinput['f']
    l = BPMinput['l']
    
    rho = BPMinput['rho']
    mu = BPMinput['mu']
    
    DT_s = BPMinput['DT_s']
    DT_p = BPMinput['DT_p']
    BT_p = BPMinput['BT_p']

    #=========================== Variable expansion ===========================
    # TE_thick = csdl.expand(h, target_shape, 'ij->aijcd') # symbol 'h': temp. value
    # slope_angle = csdl.expand(Psi, target_shape, 'ij->aijbc') # symbol 'Psi': temp.value
    # TE_thick = h   # h : size_check is needed 
    # slope_angle = Psi #Psi
    # slope_angle0 = 0.
    # slope_angle14 = 14.
    
    # Computing Strouhal number
    St_tprime = (f*TE_thick)/u   #(f*TE_thick + 1e-7)/u
    DT_avg = (DT_p + DT_s)/2
    hDel_avg = TE_thick/DT_avg   #param
    hDel_avg_prime = 6.724*(hDel_avg**2) - 4.019*(hDel_avg) + 1.107
 
    # Model G4 : eq. 74
    f1_g4 = 17.5*csdl.log(hDel_avg, 10) + 157.5 - 1.114*slope_angle
    f2_g4 = 169.7 - 1.114*slope_angle
    f_list_g4 = [f1_g4, f2_g4]
    G4 = switch_func(hDel_avg, f_list_g4, [5.])
    
    #==================== G5_14, G5_0 need to check to HJ =====================
    G5_14 = BLUNT_G5(TE_thick, slope_angle, hDel_avg, St_tprime, DT_avg)   
    G5_0 = BLUNT_G5(TE_thick, slope_angle, hDel_avg_prime, St_tprime, DT_avg)
    #==========================================================================

    # Final G5 with interpolation
    G5 = G5_0 + 0.0714*slope_angle*(G5_14 - G5_0)
    
    # transformed observer information
    S = obs_info['S']
    dh = obs_info['dh']
    
    # total BLUNT spl
    log_func = (TE_thick*(sectional_mach**5.5)*l*dh)/(S**2)
    spl_BLUNT = 10.*csdl.log(log_func, 10) + G4 + G5
    
    return spl_BLUNT


def BLUNT_G5(TE_thick, slope_angle, hDel_avg, St_tprime, DT_avg):
 
     # Model St_triple_prime: eq. 72
     f1_st = (0.212 - 0.0045*slope_angle)/(1 + (0.235*hDel_avg**(-1)) - (0.0132*(hDel_avg)**(-2)))
     f2_st = 0.1*hDel_avg + 0.095 - 0.00243*slope_angle
     f_list_Sttpr = [f1_st, f2_st]
     Sttpr_peack = switch_func(0.2, f_list_Sttpr, [hDel_avg])
     Sttpr_peack = (Sttpr_peack**2)**0.5    # This Strohal # frequently retuns negative value, even U is absolute.
     
     # Model eta : eq. 77
     eta = csdl.log(St_tprime/Sttpr_peack, 10)
     
     # Model mu : eq. 78
     f1_mu = (0.1221*hDel_avg)/hDel_avg
     f2_mu = -0.2175*hDel_avg + 0.1755
     f3_mu = -0.0308*hDel_avg + 0.0596
     f4_mu = (0.0242*hDel_avg)/hDel_avg
     f_list_mu = [f1_mu, f2_mu, f3_mu, f4_mu]
     bounds_list_mu = [0.25, 0.62, 1.15]
     mu = switch_func(hDel_avg, f_list_mu, bounds_list_mu, scale = 100.)  #
     
     # Model m : eq. 79
     f1_m = (0*hDel_avg)/hDel_avg
     f2_m = 68.724*hDel_avg - 1.35
     f3_m = 308.475*hDel_avg - 121.23
     f4_m = 224.811*hDel_avg - 69.35
     f5_m = 1583.28*hDel_avg - 1631.59
     f6_m = (268.344*hDel_avg)/hDel_avg
     f_list_m = [f1_m, f2_m, f3_m, f4_m, f5_m, f6_m]
     bounds_list_m = [0.02, 0.5, 0.62, 1.15, 1.2]
     m = switch_func(hDel_avg, f_list_m, bounds_list_m, scale = 100.)    #
     
     # Model eta0 : eq. 80
     eta0 = (-1)*((((m**2)*(mu**4))/(6.25 + ((m**2)*(mu**2))))**(0.5))
     
     # Model k : eq. 81
     k = 2.5*((1 - ((eta0/mu)**2))**0.5) - 2.5 - m*eta0
     
     # Model initial G5 : eq.  / depends on slope anlge for interpolation
     f1_g5 = m*eta + k
     f2_g5 = 2.5*(((((1 - (eta/mu)**2))**2)**0.5)**0.5) - 2.5    #none
     f3_g5 = (((1.5625 - 1194.99*(eta**2))**2)**0.5)**0.5 - 1.25 #none
     f4_g5 = -155.543*eta + 4.375
     f_list_g5 = [f1_g5, f2_g5, f3_g5, f4_g5]
     bounds_list_g5 = [eta0, 0, 0.03616]
     G5_temp = switch_func(eta, f_list_g5, bounds_list_g5, scale = 100.)
    
     # G5_temp = switch_func(eta, f_list_g5, bounds_list_g5)
     
     return G5_temp
 

def LBLVS(BPMinput, obs_info, num_observers, num_nodes)  :  
    
    a_star = BPMinput['a_star']
    sectional_mach = BPMinput['sectional_mach']
    Rc = BPMinput['Rc']
    u = BPMinput['u']
    f = BPMinput['f']
    l = BPMinput['l']
    
    rho = BPMinput['rho']
    mu = BPMinput['mu']
    
    DT_s = BPMinput['DT_s']
    DT_p = BPMinput['DT_p']
    BT_p = BPMinput['BT_p']
    
    # =========== 
    St_prime = f*BT_p/u   ##(f*BT_p)*(u + 1e-7) 
    
    # Model St1_prime : eq. 55
    f1 = (0.18*Rc)/Rc
    f2 = 0.001756*(Rc**0.3931)
    f3 = (0.28*Rc)/Rc
    f_list_Stpr1 = [f1, f2, f3]
    bounds_list_Stpr1 = [130000, 400000]
    St1_prime = switch_func(Rc, f_list_Stpr1, bounds_list_Stpr1)
    
    # eq. 56
    St_prime_peack = St1_prime*(10**(-0.04*a_star))
    
    # Model G1(e) : eq. 57
    # e = csdl.Variable(shape=target_shape, value=1e-7)   # temp. for verification
    e = St_prime/St_prime_peack
    
    f1_g1 = 39.8*csdl.log(e, 10) - 11.12
    f2_g1 = 98.409*csdl.log(e, 10) + 2.
    f3_g1 = (((2.484 - 506.25*(csdl.log(e, 10)**2))**2)**0.5)**0.5 - 5.076   # check
    f4_g1 = 2. - 98.409*csdl.log(e, 10)
    f5_g1 = -39.8*csdl.log(e, 10) - 11.12
    f_list_g1 = [f1_g1, f2_g1, f3_g1, f4_g1, f5_g1]
    bounds_list_g1 = [0.5974, 0.8545, 1.17, 1.674]
    G1 = switch_func(e, f_list_g1, bounds_list_g1)   #Q: TypeError: Value must be a numpy array, float or int. Type 'tuple' given/ checked but, tuple has not be given
    
    # reference Re : eq. 59
    f1_Rc0 = csdl.power(10, (0.215*a_star + 4.978))
    f2_Rc0 = csdl.power(10, (0.120*a_star + 5.263))
    f_list_Rc0 = [f1_Rc0, f2_Rc0]
    Rc0 = switch_func(a_star, f_list_Rc0, [3.])
    
    # Model G2(d)
    d = Rc/Rc0
    
    f1_g2 = 77.852*csdl.log(d, 10) + 15.328
    f2_g2 = 65.188*csdl.log(d, 10) + 9.125
    f3_g2 = -114.052*(csdl.log(d, 10)**2)
    f4_g2 = -65.188*csdl.log(d, 10) + 9.125
    f5_g2 = -77.852*csdl.log(d, 10) + 15.328
    f_list_g2 = [f1_g2, f2_g2, f3_g2, f4_g2, f5_g2]
    bounds_list_g2 = [0.3237, 0.5689, 1.7579, 3.0889]
    G2 = switch_func(d, f_list_g2, bounds_list_g2)
    
    # Model G3(a_star) : eq. 60
    G3 = 171.04 - 3.03*a_star
    
    # transformed observer information
    S = obs_info['S']
    dh = obs_info['dh']
    
    # Total spl for LBLVS
    log_func = (BT_p*(sectional_mach**5)*l*dh)/ (S**2)
    spl_LBLVS = 10.*csdl.log(log_func, 10) + G1 + G2 + G3
   
    return spl_LBLVS

    


import numpy as np
import csdl_alpha as csdl

from BPMsplModel import switch_func

recorder = csdl.Recorder(inline = True)
recorder.start()

class BPMsplModel():

    def BPMsplModel_TBLTE(observer_data, input_data, SteadyObserver, num_radial, num_tangential, num_nodes = 1):
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
        
        #================= BPM SPL inputs from BPM_model.py ====================
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
        
        #========================== Variable expansion =========================
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
        
        # rc = csdl.Variable('Rc', target_shape)  ##Q: initially defined as variable, but value  = none?
        rc = u*csdl.expand(chord_profile, target_shape, 'ij -> ijab')/csdl.expand(nu,target_shape) + 1e-7
        Rsp = u*boundaryP/nu + 1e-7  #old: Rsp = csdl.Variable('Rdp', target_shape)
                        
        sectional_mach = u/c0
        
        #==================== Computing St (Strouhal numbers) ==================
        sts = (f*boundaryS_disp)/(u + 1e-7)     ## old: resister_output -> directly write equation
        stp = (f*boundaryP_disp)/(u + 1e-7)
        st1 = 0.02*((sectional_mach + 1e-7)**(-0.6))
        
        # Model St2 : eq. 34
        f_1 = st1*1
        f_2 = st1*csdl.power(10, 0.0054*((AOA-1.33)**2))
        f_3 = st1*4.72
        funcs_listst2 = [f_1, f_2, f_3]
        bounds_listst2 = [1.33, 12.5]
        st2 = switch_func(AOA, funcs_listst2, bounds_listst2)
        
        # Model bar(St1) : eq. 33
        stPROM = (st1+st2)/2
        St = csdl.max(sts, stp, rho = 1000)  #check
        St_peack = csdl.max(st1, st2, stPROM, rhod=1000) #check
        
        #========================== Computing coeff. A =========================
        a = csdl.log(((St/St_peack+1e-7)**2)**0.5, 10) ##Q: why (**2)**0.5 = 1, due to absolute value?
        
        # Model A : eq. 35
        f1b = (((67.552 - 886.778*(a**2))**2)**0.5)**(0.5) - 8.219
        f2b = (-32.665*a) + 3.981
        f3b = (-142.795*(a**3)) + (103.656*(a**2)) - (57.757*a) + 6.006
        f_list_b = [f1b, f2b, f3b]
        bounds_list_b = [0.204, 0.244]
        aMin = switch_func(a, f_list_b, bounds_list_b)
     
        # Model A : eq. 36
        f1c = ((((67.552 - 886.788*(a**2))**2)**0.5)**0.5) - 8.219
        f1c = ((((67.552 - 886.788 * a**2)**2)**0.5) ** 0.5) - 8.219
        f2c = (-15.901 * a) + 1.098
        f3c =  (-4.669 * a*3) + (3.491 * a*2) - (16.699 * a) + 1.149
        f_list_c = [f1c, f2c, f3c]
        bounds_list_c = [0.13, 0.321]
        aMax = switch_func(a, f_list_c, bounds_list_c)
     
        # === a0 ====
        # Model A : eq. 38
        f1a = (rc+1e-7)*0.57/(rc+1e-7) ##Q : (rc+1e-7)/(rc+1e-7) = 1
        f2a = (-9.57*(10**(-13)))*((rc - (857000))**2) + 1.13
        f3a = (1.13 * rc)/rc
        f_list_a =[f1a, f2a, f3a]
        bounds_list_a = [95200, 857000]
        a0 = switch_func(rc, f_list_a, bounds_list_a)
        
        # Model A : eq. 35 for a0
        f1a0 = ((((67.552 - 886.788 * (a0**2))**2)**0.5) ** 0.5) - 8.219
        f2a0 = (-32.665 * a0) + 3.981
        f3a0 = (-142.795 * (a0**3)) + (103.656 * (a0**2)) - (57.757 * a0) + 6.006
        f_list_a0 = [f1a0, f2a0, f3a0]
        bounds_list_a0 = [0.204, 0.244]
        a0Min = switch_func(a0, f_list_a0, bounds_list_a0)
        
        # Model A : eq. 36 for a0
        f1c0 = ((((67.552 - 886.788 * (a0**2))**2)**0.5) ** 0.5) - 8.219
        f2c0 = (-15.901 * a0) + 1.098
        f3c0 = (-4.669 * a0*3) + (3.491 * a0*2) - (16.699 * a0) + 1.149
        f_list_c0 = [f1c0, f2c0, f3c0]
        bounds_list_c0 = [0.13, 0.321]
        a0Max = switch_func(a0, f_list_c0, bounds_list_c0)
    
        # Model Ar : eq. 39
        AR_a0 = (-20 - a0Min) / (a0Max - a0Min)
        # Model A(a) = eu. 40
        A_a = aMin + (AR_a0 * (aMax - aMin))
     
        #========================== Computing coeff. B =========================
        # ==== b ====
        # Model b : eq. 43
        b = csdl.power(10, ((sts/(st2+1e-7)**2)**0.5))
        
        # Model B_min(b) : eq. 41
        f1 = ((((16.888 - (886.788*(b**2)))**2)**0.5)**0.5) - 4.109   ##Q: why b*b instead of b**2?
        f2 = (83.607*(-1)*b) + 8.138
        f3 = (817.81*(-1)*(b**3)) + (355.21*(b**2)) - (135.024*b) + 10.619 
        funcs_listbMin = [f1, f2, f3]
        bounds_listbMin = [0.13, 0.145]
        bMin = switch_func(b, funcs_listbMin, bounds_listbMin)
    
        # Model B_max(b) : eq. 42
        f4 = ((((16.888 - (886.788*(b**2)))**2)**0.5)**0.5) - 4.109
        f5 = 1.854 - (31.33*b)
        f6 = (80.541*(-1)*(b**3)) + (44.174*(b**2)) - (39.381*b) + 2.344
        funcs_listbMax = [f4, f5, f6]
        bounds_listbMax = [0.10, 0.187]
        bMax = switch_func(b, funcs_listbMax, bounds_listbMax)
        
        # ==== b0 ====
        # Model B
        f7 = (rc*0.3)/rc
        f8 = (-4.48*(10**(-13))) * ((rc-(8.57*(10**5)))**2) + 0.56
        f9 = (0.56*rc)/rc 
        funcs_listb0 = [f7, f8, f9]
        bounds_listb0 = [95200.0, 857000.0]
        b0 = switch_func(rc, funcs_listb0, bounds_listb0)
       
        # Model B_min(b0) for eq. 45
        f10 = ((((16.888-(886.788*(b0**2)))**2)**0.5)**0.5) - 4.109
        f11 = (83.607*(-1)*b0) + 8.138
        f12 = (817.81*(-1)(b0**3)) + (355.21*(b0**2)) - (135.024*b0) + 10.619
        funcs_listb0Min = [f10, f11, f12]
        bounds_listb0Min = [0.13, 0.145]
        b0Min = switch_func(b, funcs_listb0Min, bounds_listb0Min)
    
        # Model B_max(b0) for eq. 45
        f13 = ((((16.888-(886.788*(b0**2)))**2)**0.5)**0.5) - 4.109
        f14 = 1.854 - (31.33*b0)
        f15 = (80.541*(-1)*(b0**3)) + (44.174*(b0**2)) - (39.381*b0) + 2.344
        funcs_listb0Max = [f13, f14, f15]
        bounds_listb0Max = [0.10, 0.187]
        b0Max = switch_func(b, funcs_listb0Max, bounds_listb0Max)
     
        # Model B_R(b0) : eq. 45
        BR = (-20 - b0Min)/((b0Max) - (b0Min))
        
        # Model B(b) : eq. 46
        Bb = bMin + (BR*(bMax - bMin))
    
        # ========================== Computing coeff. K ========================
        # Model K1 : eq. 47
        f1k = -4.31 * csdl.log((rc+1e-7), 10) + 156.3
        f2k = -9.0 * csdl.log((rc+1e-7), 10) + 181.6
        f3k = (128.5*rc)/rc
        f_list_k = [f1k, f2k, f3k]
        bounds_list_k = [247000, 800000]
        k1 = switch_func(rc, f_list_k, bounds_list_k)
        
        # Model delta_K1 : eq. 48
        f1ak = AOA * (1.43 * csdl.log((Rsp + 1e-7), 10) - 5.29)
        f2ak = Rsp*0
        f_list_ak = [f1ak, f2ak]
        bounds_list_ak = [5000]
        ak1 = switch_func(Rsp, f_list_ak, bounds_list_ak)
        
        # gamma, gamma0, beta, beta0: eq. 50
        y = (27.094 * sectional_mach) + 3.31  # y = gamma
        y0 = (23.43 * sectional_mach) + 4.651
        betha = (72.65 * sectional_mach) + 10.74
        betha0 = (-34.19 * sectional_mach) - 13.82
        
        # Model K2 : eq. 49
        f1k2 = (-1000*betha)/betha
        f2k2 = (((((betha**2)-(((betha/y)**2)*((AOA-y0)**2)))**2)**0.5)**0.5) + betha0
        f3k2 = (-12*betha)/betha
        f_list_k2 =[k1 + f1k2, k1 + f2k2, k1 + f3k2]
        bounds_list_k2 = [y0 - y, y0 + y]
        k2 = switch_func(AOA, f_list_k2, bounds_list_k2)
    
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
        
        # Noise compoents
        log_func1 = (boundaryP_disp*(sectional_mach**5)*l*dh)/(S**2) + 1e-7
        log_func2 = (boundaryS_disp*(sectional_mach**5)*l*dh)/(S**2) + 1e-7
    
        splp = 10.*csdl.log(log_func1) + A_a + (k1 - 3) + ak1
        spls = 10.*csdl.log(log_func2) + A_a + (k1 - 3) 
        spla = 10.*csdl.log(log_func2) + Bb + k2
        
        # Total noise
        spl_TOT = 10.*csdl.log(csdl.power(10, spla/10.) + csdl.power(10, spls/10.) + csdl.power(10, splp/10.))
    
        # ============================== Ref. HJ ===============================
        # Spp_bar = csdl.exp_a(10., SPLTOT/10.) # shape is (num_nodes, num_observers, num_radial, num_azim)
        # Mr = u / csdl.expand(
        #     self.declare_variable('speed_of_sound', shape=(num_nodes,)),
        #     (num_nodes, num_observers, num_radial, num_azim),
        #     'i->iab'
        # )
    
        # W = 1/(1 + Mr*x_r/re) # shape is (num_nodes, num_observers, num_radial, num_azim)
        # Spp = csdl.sum(Spp_bar/(W**2), axes=(3,)) * 2*np.pi/num_azim/(2*np.pi) # (num_nodes, num_observers, num_radial)
    
        # finalSPL = 10*csdl.log(csdl.sum(Spp, axes=(2,)))
        # self.register_output(f'{component_name}_broadband_spl', finalSPL) # SHAPE IS (num_nodes, num_observers)
        # # endregion
        # import numpy as np
        # import csdl_alpha as csdl

    def BPMsplModel_LBLVS():
     
        
        #============================ Define input =============================
        sectional_mach = u/c0

        # mach = csdl.expand(csdl.Variable(value = M), out_shape = target_shape)
        
        boundary_thickP = csdl.expand(csdl.Variable(value = , shape = (num_nodes, num_radial)), target_shape, 'ij -> ')
        rpm = csdl.expand(csdl.Variable(shape = num_nodes,), target_shape, 'i -> ')
        
        f = num_bades * rpm / 60  ##Q : temp. value
        AOA = csdl.expand(csdl.Variable(value = a_star, shape = (num_radial, )), target_shape, 'i -> ')
        
        rc = csdl.Variable(target_shape)  ##Q: initially defined as variable, but value  = none?
        Rsp = csdl.Variable(target_shape) 
        #========================== Variable expansion =========================
        target_shape = (num_nodes, num_observers, num_radial, num_azim)

        
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
        log_func = (boundary_thickP*(sectional_mach**5)*l*dh)/ (S**2)
        spl_LBLVS = 10.*log(log_func, 10) + G1 + G2 + G3
    
    
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
            

        
    def convection_adjustment(self, S, x, y, z, c0, num_nodes, num_observers, num_radial, num_azim):
        # mechC, theta, psi = convection_adjustment(S, x_r, y_r, z_r,c0, num_nodes, num_observers, num_radial, num_azim)
        position_vec = csdl.Variable()  #shape or value must be provided.
        V_vec = csdl.Variable() #shape or value must be provided.
        
        x_pos = csdl.expand(x, (num_nodes, num_observers, num_radial, num_azim, 1), 'ijkl -> ikjla')
        y_pos = csdl.expand(y, (num_nodes, num_observers, num_radial, num_azim, 1), 'ijkl -> ikjla')
        z_pos = csdl.expand(z, (num_nodes, num_observers, num_radial, num_azim, 1), 'ijkl -> ikjla')
        position_vec[:,:,:,:,0] = x_pos
        position_vec[:,:,:,:,1] = y_pos
        position_vec[:,:,:,:,2] = z_pos
        
        Vx = csdl.expand(csdl.Variable('Vx', shape=(num_nodes,)), (num_nodes, num_observers, num_radial, num_azim, 1), 'i -> iabcd')
        Vy = csdl.expand(csdl.Variable('Vy', shape=(num_nodes,)), (num_nodes, num_observers, num_radial, num_azim, 1), 'i -> iabcd')
        Vz = csdl.expand(csdl.Variable('Vz', shape=(num_nodes,)), (num_nodes, num_observers, num_radial, num_azim, 1), 'i -> iabcd')
        V_vec[:,:,:,:,0] = Vx
        V_vec[:,:,:,:,1] = Vy
        V_vec[:,:,:,:,2] = Vz
        V_conv = csdl.product(V_vec, position_vec, axes = 4)/S # check
        machC = V_conv / c0 # check : shape of 'c0'
        
        x_pos_re = csdl.reshape(x_pos, (num_nodes, num_observers, num_radial, num_azim))
        y_pos_re = csdl.reshape(y_pos, (num_nodes, num_observers, num_radial, num_azim))
        z_pos_re = csdl.reshape(z_pos, (num_nodes, num_observers, num_radial, num_azim))
        x_z_mag = (x_pos_re**2 + z_pos_re**2)**0.5
        theta = csdl.arccos(x/S)
        y_z_mag = (y_pos_re**2 + z_pos_re**2)**0.5
        psi = csdl.arccos(y_pos_re/y_z_mag)
        
        # self.resister_output('theta_dumy', theta) #old
        
        return machC, theta, psi
        
            

    
                
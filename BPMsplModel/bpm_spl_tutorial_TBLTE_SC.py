import numpy as np
import csdl_alpha as csdl

from lsdo_acoustics.core.models.observer_location_model import SteadyObserverLocationModel
from lsdo_acoustics.core.models.broadband.BPM.bpm_spl_model import BPMSPLModel

from lsdo_acoustics.utils.atmosphere_model import AtmosphereModel

class BPMsplModel():
    def __init__(self, observer_data, input_data, num_azim, num_nodes = 1):
        self.observer_data = observer_data
        self.input_data = input_data
        self.num_azim = num_azim
        self.num_nodes = num_nodes
        
        # self.num_blades = num_blades
        # self.num_radial = num_radial
        # self.num_observers = num_observers

    
    def BPMsplModel_TBLTE(self):
        #================================ Input ================================
        num_nodes = self.num_nodes
        observer_data = self.observer_data
        num_blades = self.input_data['num_blades']
        rpm = self.input_data['rpm']
        
        num_radial = self.input_data['num_radial']
        num_azim = self.num_azim
        num_observers = self.observer_data['num_observers']
        
        propeller_radius = self.input_data['radius']
        chord_profile = csdl.Variable(value = chord_profile, shape = (num_radial,1))
        # twist_profile = csdl.Variable(value = 0.*np.pi/180, shape = (num_radial,1))
        # thrust_dir = csdl.Variable(valuae = np.array([0., 0., 1.]), shape = (3,))
        
        chord = self.input_data['chord']
        chord_length = csdl.reshape(csdl.norm(chord, axes = 1), (num_radial, 1))
        
        #================= BPM SPL inputs from BPM_model.py ====================
        # rho = density
        # mu = dynamic_viscosity
        # nu = nu
        non_dim_r = csdl.Variable(value = np.linspace(0.2, 1., num_radial))
        non_dim_rad_exp = csdl.expand(non_dim_r, (num_nodes, num_radial), 'i->ai')
        
        # rpm = csdl.expand(rpm, (num_nodes, num_radial))
        
        U = non_dim_rad_exp* (2*np.pi/60.) * rpm * csdl.expand(propeller_radius, (num_nodes, num_radial))
                
        delta_P = csdl.Variable(value = 3.1690e-4, shape = (num_nodes, num_radial))
        delta_S = csdl.Varaible(value = 3.1690e-4,shape = (num_nodes, num_radial))
        
        a_CL0 = csdl.Variable(value = 0, shape = (num_radial, 1))
        aoa = csdl.Variable(value = 0., shape = (num_radial, 1))
        a_star = aoa - a_CL0
        
        #========================== Variable expansion =========================
        target_shape = (num_nodes, num_observers, num_radial, num_azim)
        # mach = csdl.expand(csdl.Variable(value = M), target_shape)
        # visc = csdl.expand(csdl.Variable('nu'), target_shape)
        u = csdl.expand(U, target_shape, 'ij -> iajb')
        l = csdl.expand(propeller_radius/num_radial, target_shape)
        S = csdl.expand(S_r, target_shape)
        c0 = csdl.expand(0., target_shape)   # c0 = sound_of_speed
    
        boundaryP = csdl.expand(delta_P, target_shape, 'ij -> iajc')
        boundaryS = csdl.expand(delta_S, target_shape, 'ij -> iajc')
        rpm = csdl.expand(rpm, target_shape, 'i -> iabc')
        
        f = num_blades * rpm / 60  ##Q : temp. value
        AOA = csdl.expand(a_star, target_shape, 'i -> abic')
        
        # rc = csdl.Variable('Rc', target_shape)  ##Q: initially defined as variable, but value  = none?
        rc = u*csdl.expand(chord_profile, target_shape, 'ij -> ijab')/csdl.expand(nu,target_shape) + 1e-7
        Rsp = u*boundaryP/nu + 1e-7  #old: Rsp = csdl.Variable('Rdp', target_shape)
                        
        sectional_mach = u/c0
        
        #==================== Computing St (Strouhal numbers) ==================
        sts = (f*boundaryS)/(u + 1e-7)     ## old: resister_output -> directly write equation
        stp = (f*boundaryP)/(u + 1e-7)
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
        f1 = ((((16.888 - (886.788*b*b))**2)**0.5)**0.5) - 4.109   ##Q: why b*b instead of b**2?
        f2 = (83.607*(-1)*b) + 8.138
        f3 = (817.81*(-1)*b*b*b) + (355.21*b*b) - (135.024*b) + 10.619 
        funcs_listbMin = [f1, f2, f3]
        bounds_listbMin = [0.13, 0.145]
        bMin = switch_func(b, funcs_listbMin, bounds_listbMin)
    
        # Model B_max(b) : eq. 42
        f4 = ((((16.888 - (886.788*b*b))**2)**0.5)**0.5) - 4.109
        f5 = 1.854 - (31.33*b)
        f6 = (80.541*(-1)*b*b*b) + (44.174*b*b) - (39.381*b) + 2.344
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
        f10 = ((((16.888-(886.788*b0*b0))**2)**0.5)**0.5) - 4.109
        f11 = (83.607*(-1)*b0) + 8.138
        f12 = (817.81*(-1)*b0*b0*b0) + (355.21*b0*b0) - (135.024*b0) + 10.619
        funcs_listb0Min = [f10, f11, f12]
        bounds_listb0Min = [0.13, 0.145]
        b0Min = switch_func(b, funcs_listb0Min, bounds_listb0Min)
    
        # Model B_max(b0) for eq. 45
        f13 = ((((16.888-(886.788*b0*b0))**2)**0.5)**0.5) - 4.109
        f14 = 1.854 - (31.33*b0)
        f15 = (80.541*(-1)*b0*b0*b0) + (44.174*b0*b0) - (39.381*b0) + 2.344
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
    
    
        # mechC, theta, psi = convection_adjustment(S, x_r, y_r, z_r, num_nodes, num_observers, num_radial, num_azim)
        dh= (((2*(csdl.sin(theta/2))**2)*((csdl.sin(psi))**2))/((1+(sectional_mach*csdl.cos(theta)))*(1+(sectional_mach-machC)*csdl.cos(theta))**2))   #EQ B1
        
        # Noise compoents
        log_func1 = (boundaryP*(sectional_mach**5)*l*dh)/(S**2) + 1e-7
        log_func2 = (boundaryS*(sectional_mach**5)*l*dh)/(S**2) + 1e-7
    
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
    
        
# =============================================================================
#         # DIRECTIVITY COMPUTATION (APPENDIX: B1)
#         def directivity_computation():
# =============================================================================
            
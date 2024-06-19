import numpy as np
import csdl_alpha as csdl

"""
Initial tutorial : gl_spl_model (05/12/24)
"""

class GLSPLModel():
    
    def __init__(self, num_nodes, name, num_observers, num_blades, num_radial, freq_band):
        csdl.check_parameter(num_nodes, 'num_nodes', types=(float, int))
        csdl.check_parameter(name, default=None, types=str, allow_none=True)
        csdl.check_parameter(num_observers, 'num_observers', types=(float,int))
        csdl.check_parameter(num_blades, 'num_blades', types=(float,int))
        csdl.check_parameter(num_radial,'num_radial', types=(float,int))
        csdl.check_parameter(freq_band, default = np.array(
                 [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
                  500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000, 12500, 16000, 20000,
                  25000, 31500, 40000, 50000, 63000 # additional ones used by Hyunjune
                  ]
        ))
        
        # assign parameters to the class
        self.num_nodes = num_nodes
        self.name = name
        self.num_observers = num_observers
        self.num_blades =num_blades
        self.num_radial = num_radial
        self.freq_band = freq_band
        
    def gl_spl(self, inputs1: csdl.VariableGroup, inputs2: csdl.VariableGroup):
        num_nodes = self.parameters['num_nodes']
        num_observers = self.parameters['num_observers']
        model_name = self.parameters['name']
        
        B = self.parameters['num_blades']
        num_radial = self.parameters['num_radial']
        
        freq_band = self.parameters['freq_band']
        frequency_band = self.creat_input('frequency_band',freq_band)
        num_freq_band = len(freq_band)   
        

        # Inputs to model:
        CT = self.declare_input('CT', inputs1.CT)   
        theta_0 = self.declare_input('theta_0', inputs1.theta_0)
        R = self.declare_input('R', inputs1.R)
        S = self.declare_input('S', inputs1.S)
        chord_profile = self.declare_input('chord_profile', inputs1.chord_profile)
     
        dr = self.declare_input('dr', inputs2.dr)
        rpm = self.declare_input('rpm', inputs2.rpm)
        a = self.declare_input('speed_of_sound',shape=(num_nodes,), value = 343)
        
        V_aircraft = self.declare_input('V_aircraft', inputs2.V_aircraft)
        V_inf = csdl.pnorm(V_aircraft, axis=1)
        
        omega = rpm*2*np.pi/60.
        V_tip = omega*R
        A_b = csdl.sum(chord_profile)*dr
        sigma = A_b*B/(np.pi*R**2)
        c_sw = sigma*np.pi*R/B
        
        V_t = V_tip
        M_t = V_t/a
        St = frequency_band*c_sw
        
        f0 = csdl.log10(V_t**7.84)*10.
        f1 = sigma
        f2 = 0.9*M_t*sigma*(M_t+3.82)
        f3 = 1. # NOT USED
        f4 = 1. # NOT USED
        f5 = -2.*M_t**2+2.06
        f6 = -CT*M_t*(CT-csdl.sin((theta_0**2)**0.5)+2.06)+1.
        f7 = CT
        f8 = 4.97*CT*csdl.sin((theta_0**2)**0.5)*(1.5*S/R*M_t - (S/R) + 15.)

        num = f0*(St-(f1*csdl.log10(CT) + f2*csdl.log10(sigma)))**0.6
        den_1 = csdl.exp_a(
            f3*(St-(f1*csdl.log10(CT)+f2*csdl.log10(sigma)) + f5),
            f6
        )
        den_2 = csdl.exp_a(
            f7*(St-(f1*csdl.log10(CT)+f2*csdl.log10(sigma))),
            f8
        )
        
        SPL_1_3 = num/(den_1 + den_2)
        
        return SPL_1_3
        
        # def obs_distance



        
        
        
                                    
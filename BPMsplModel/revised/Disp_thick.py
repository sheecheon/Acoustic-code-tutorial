import numpy as np
import csdl_alpha as csdl
from revised.csdl_switch import switch_func

def Disp_thick(a_star, Rc, chord):
       # Untripped boundary layer @ zero angle of attack (zero lift for NACA0012)
       BT_0 = chord*csdl.power(10, (1.6569 - 0.9045*csdl.log(Rc, 10) + 0.0596*(csdl.log(Rc, 10))**2)) #Boundary layer thickness [m]
       DT_0 = chord*csdl.power(10, (3.0187 - 1.5397*csdl.log(Rc, 10) + 0.1059*(csdl.log(Rc, 10))**2)) #Displacement thickness [m]

       # Heavily tripped boundary layer @ zero angle of attack (zero lift for NACA0012)
       BT_0_trip = chord*csdl.power(10, (1.892 - 0.9045*csdl.log(Rc, 10) + 0.0596*(csdl.log(Rc, 10))**2)) 
       
       f1_DT_trip = chord*0.0601*Rc**(-0.114)
       f2_DT_trip = chord*csdl.power(10, (3.411 - 1.5397*csdl.log(Rc, 10) + 0.1059*(csdl.log(Rc, 10))**2))
       funcs_DT_trip = [f1_DT_trip, f2_DT_trip]
       DT_0_trip = switch_func(Rc, funcs_DT_trip, [0.3*(10**6)])   # Q: Rc<=0.3*10^6 or 0.3*10^6<Rc : switch function properly?

       # Pressure side boundary layers (for both tripped and untripped BL)
       BT_p_trip = BT_0_trip*csdl.power(10, (-0.04175*a_star + 0.001060*a_star**2)) # Pressure side boundary layer thickness [m]
       BT_p_untrip = BT_0*csdl.power(10, (-0.04175*a_star + 0.001060*a_star**2)) # Pressure side boundary layer thickness [m]

       DT_p_trip = DT_0_trip*csdl.power(10, (-0.04320*a_star + 0.001130*a_star**2)) # Pressure side displacement thickness [m]
       DT_p_untrip = DT_0*csdl.power(10, (-0.04320*a_star + 0.001130*a_star**2)) # Pressure side displacement thickness [m]

       # Suction side boundary layer (for tripped BL)
       f1_DT_s_trip = DT_0_trip*csdl.power(10, (0.0679*a_star)) # Suction side displacement thickness [m]
       f2_DT_s_trip = DT_0_trip*0.381*csdl.power(10, (0.1516*a_star)) # Suction side displacement thickness [m]
       f3_DT_s_trip = DT_0_trip*14.296*csdl.power(10, (0.0258*a_star))  # Suction side displacement thickness [m]
       funcs_DT_s_trip = [f1_DT_s_trip, f2_DT_s_trip, f3_DT_s_trip]
       bounds_DT_s_trip = [5, 12.5]
       DT_s_trip = switch_func(a_star, funcs_DT_s_trip, bounds_DT_s_trip)

       # Suction side boundary layer (for untripped BL)
       f1_DT_s_untrip = DT_0*csdl.power(10, (0.0679*a_star)) # Suction side displacement thickness [m]
       f1_DT_s_untrip = DT_0*0.0162*csdl.power(10, (0.3066*a_star)) # Suction side displacement thickness [m]
       f3_DT_s_untrip = DT_0*52.42*csdl.power(10, (0.0258*a_star)) # Suction side displacement thickness [m]
       funcs_DT_s_untrip = [f1_DT_s_untrip, f1_DT_s_untrip, f3_DT_s_untrip]
       bounds_DT_s_untrip = [7.5, 12.5]
       DT_s_untrip = switch_func(a_star, funcs_DT_s_untrip, bounds_DT_s_untrip)

       # Current: Only 'empirical trip' condition is used
       return DT_s_trip, DT_p_trip, BT_p_trip
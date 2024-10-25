from threading import local
import numpy as np
import csdl_alpha as csdl
from lsdo_acoustics.core.acoustics import Acoustics
from revised.BPM_model import BPMVariableGroup, BPM_model
from revised.A_weighting_function import A_weighting_function
import time
import pickle
from scipy import io

# =============================================================================
# class DummyMesh(object):
#     def __init__(self, num_radial, num_tangential):
#         self.parameters = {
#             'num_radial': num_radial,
#             'num_tangential': num_tangential,
#             'mesh_units': 'm'
#         }
# 
# # observer data
# a = Acoustics(aircraft_position=np.array([0., 0., 0.,]))
# 
# obs_radius = 1.5
# num_observers = 37
# theta = np.linspace(0, np.pi, num_observers)
# z = obs_radius * np.cos(theta)
# x = obs_radius * np.sin(theta)
# 
# obs_position_array = np.zeros((num_observers, 3))
# obs_position_array[:,0] = x
# obs_position_array[:,2] = z
# 
# for i in range(num_observers):
#     a.add_observer(
#         name=f'obs_{i}',
#         obs_position=obs_position_array[i,:],
#         time_vector=np.array([0.])
#     )
# observer_data = a.assemble_observers()
# =============================================================================
recorder = csdl.Recorder(inline = True)
recorder.start()

with open('EdgewiseInput_SUI', 'rb') as f:
    edgewise_input = pickle.load(f)
    
# ======================= Blade condition : 'SUI' ==========================
flight_condition = 'edgewise'
outband = 'one third'
x = 0.
y = 2.2755
z = -2.7118
num_observers = 1
observer_data = {'x': x,
                 'y': y,
                 'z': z,
                 'num_observers': num_observers
                 }

num_blades = 2
RPM = 4047
radius = 0.1904

M = 0.045
rho = 1.225         # air density
mu = 1.789*(1e-5)   # dynamic_viscosity
c0 = 340.3          # sound of speed
omega = RPM*2.*np.pi/60.
Vinf = M*c0   # This does not work as V_inf, currently only used for obs time computation

freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                 6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]
num_freq = len(freq)

chord = np.array([0.0207635084817145, 0.0230402481821243, 0.0257950422862420, 0.0286433015611225,
                  0.0312092825621372, 0.0337744145336160, 0.0353421734018633, 0.0365224382527332,	
                  0.0373550477817100, 0.0378608723451693, 0.0380741423668972, 0.0380834682219359,	
                  0.0379621090979672, 0.0376993582918083, 0.0372915490719629, 0.0367714372663591,	
                  0.0362274474160400, 0.0356786649715456, 0.0352709218953821, 0.0347399881060303,	
                  0.0340535999561616, 0.0332747841690797, 0.0324668557508378, 0.0314848941287359,	
                  0.0306609937425907, 0.0297414528019706, 0.0287845180464525, 0.0278460961175235,	
                  0.0269248271600174, 0.0257639205107062, 0.0247871652826624, 0.0238533755701973,	
                  0.0231095502040669, 0.0224182624433513, 0.0214853745612483, 0.0199448932720913,	
                  0.0187643702719414, 0.0174013712380026, 0.0152597826007824, 0.0117384558384340])

radial = np.linspace(0.026, 0.188, 40)
sectional_span = radial[2] - radial[1]
num_radial = len(radial)
azimuth = np.linspace(0, 2*np.pi, 20)   
num_azim = len(azimuth)
num_nodes = 1 

AOA = edgewise_input['AoA']  # 3D tensor input in 'Edgewise' condition : (num_radial, num_azim, num_blades) 
                
A_cor = 0.
a_star = AOA + A_cor
   
V0_data = edgewise_input['V0']    # 3D tensor input in 'Edgewise' condition : (num_radial, num_azim, num_blades) 
V0_init = (V0_data**2)**0.5       # To refurn abs.value

# Velocity computation
RPM =csdl.Variable(value = RPM)   # csdl.var
omega = RPM*2.*np.pi/60.
azim_exp = csdl.expand(azimuth, (num_azim, num_radial, num_blades),'i->iab')

Vr = omega*radial
Vr_exp = csdl.expand(Vr, (num_azim, num_radial, num_blades),'i->aib')
V_inf = (V0_data - Vr_exp)/csdl.cos(azim_exp)   #new V_inf value
V0 = ((Vr_exp + V_inf*csdl.cos(azim_exp))**2)**0.5

pitch = np.array([5.71827196, 9.73526282, 13.81025512, 17.0172273,  18.47316765,
                  18.0755777, 17.1715650, 16.19259423, 15.24232188, 14.33239574,
                  13.4617839, 12.5032064, 11.8446102,  11.18607406, 10.51135360,
                  9.88999153,  9.3711866,  8.89386211,  8.57064106,  8.19189033,
                  7.76509555,  7.3859008,  7.06844137,  6.70339299,  6.45105441,
                  6.24791936,  6.0390358,  5.79490773,  5.57605766,  5.36631695,
                  5.17705074,  4.9779330,  4.83475723,  4.74343221,  4.66031611,
                  4.54590395,  4.4733133,  4.42699749,  4.4035366 ,  4.39935211])

alpha = 0
sigma = 0.0712
TE_thick = 7.62e-4 # h
slope_angle = 19.  # Psi

# ======================== BPM spl computation =============================
BPM_vg = BPMVariableGroup(
    chord = chord,
    radial = radial, #R
    sectional_span = sectional_span, # R(1) - R(2) / l
    a_star = a_star, # AoAcor = 0. / a_star = AOA - AOAcor.
    pitch = pitch,
    azimuth = azimuth,
    alpha = alpha,
    RPM = RPM,
    TE_thick = TE_thick, #h
    slope_angle = slope_angle, #Psi
    free_vel = V0_init, #free-stream velocity U and V0
    freq = freq,
    speed_of_sound = c0, 
    Vinf = Vinf,
    density = rho, 
    dynamic_viscosity = mu,  #mu
    num_radial = num_radial,
    num_tangential = num_azim,   # num_tangential = num_azim
    num_freq = num_freq
    )          
           
SPL_rotor, OASPL = BPM_model(BPMVariableGroup = BPM_vg,
                             observer_data = observer_data,
                             num_blades = num_blades,
                             num_nodes = 1,
                             flight_condition = flight_condition,
                             outband = outband
                             )

print('rotor SPL is :', SPL_rotor.value)

BPM_vg2 = BPMVariableGroup(
    chord = chord,
    radial = radial, #R
    sectional_span = sectional_span, # R(1) - R(2) / l
    a_star = a_star, # AoAcor = 0. / a_star = AOA - AOAcor.
    pitch = pitch,
    azimuth = azimuth,
    alpha = alpha,
    RPM = RPM,
    TE_thick = TE_thick, #h
    slope_angle = slope_angle, #Psi
    free_vel = V0, #free-stream velocity U and V0
    freq = freq,
    speed_of_sound = c0, #c0
    Vinf = Vinf,
    density = rho,  #rho
    dynamic_viscosity = mu,  #mu
    num_radial = num_radial,
    num_tangential = num_azim,   # num_tangential = num_azim
    num_freq = num_freq
    )          
           
SPL_rotor2, OASPL2 = BPM_model(BPMVariableGroup = BPM_vg2,
                               observer_data = observer_data,
                               num_blades = num_blades,
                               num_nodes = 1,
                               flight_condition = flight_condition,
                               outband = outband
                               )
print('rotor SPL 2 is :', SPL_rotor2.value)

# =========================== A-weighting computation ===========================
SPL_rotor_A = A_weighting_function(SPL_rotor2, freq)
print('A-weighting values :', SPL_rotor_A.value)

# =========================== Derivative computation ===========================
asdf = csdl.derivative(ofs = OASPL2, wrts = RPM)
print(f'derivative value: {asdf.value}')
asdf = csdl.derivative_utils.verify_derivatives(ofs = OASPL, wrts = RPM, step_size = 1.e-6)

# asdf1 = csdl.derivative(ofs=OASPL, wrts=slope_angle)
# print(f'derivative value: {asdf.value}')
# asdf1 = csdl.derivative_utils.verify_derivatives(ofs = OASPL, wrta = slope_angle, step_size = 1.e-6)

# =================== HJ's ref. data for Verification =========================
import matplotlib.pyplot as plt

BPM_HJ_SPL_rotor = np.array([53.7052487220358, 51.8659498137181, 51.0080812978605, 51.2735514736476,	
                             52.0652058093291, 53.1080365693955, 54.3850685077111, 54.8031515905698,
                             54.4466585575157, 53.6514690757390, 52.1130113017996, 50.1159298004450,
                             47.5391159959926, 44.8409843760571, 42.7914939927355, 41.1931933956194,
                             36.6260331648658, 30.8379586348276, 26.2005273229949, 23.3852435706749,	
                             20.7540837221740, 17.9536253554758])

SPL_rotor = SPL_rotor.value.reshape(-1)
rel_error = abs((BPM_HJ_SPL_rotor - SPL_rotor))/BPM_HJ_SPL_rotor

print('Relative error of BPM with respect to Frequency :', rel_error)

SPL_rotor2 = SPL_rotor2.value.reshape(-1)
rel_error2 = abs((BPM_HJ_SPL_rotor - SPL_rotor2))/BPM_HJ_SPL_rotor
print('Relative error of BPM with respect to Frequency :', rel_error2)

plt.figure()
plt.semilogy(freq, rel_error, label="V0 : from HJ data")
plt.semilogy(freq, rel_error2, label="V0 : obtained by tangential velocitys")
plt.xlabel('Frequency [HZ]')
plt.ylabel('Relative error')
plt.legend(['V0 : from HJ data', 'V0 : obtained by tangential velocity'], loc = 'upper right')
plt.grid()
plt.show()

# # plt.legend(['SPL computation with csdl.nonlinear solver', 'SPL computation with imported observer time'])
# plt.title('Relative error of BPM model')
# plt.xlabel('Frequency [HZ]')
# plt.ylabel('Relative error')
# plt.grid()
# plt.show()

# # SPL rotor A-weighting value plo t
# SPL_rotor_A = SPL_rotor_A.value.reshape(-1)
# plt.figure()
# plt.plot(freq, SPL_rotor2)
# plt.plot(freq, SPL_rotor_A)
# plt.legend(['SPL [dB]', 'A-weighting [dBA]'])
# plt.title('Rotor SPL and A-weighting values')
# plt.xlabel('Frequency [HZ]')
# plt.ylabel('SPL values')
# plt.grid()
# plt.show()

'''
At the higher freqency (the last 3~4 terms), it shows noticeable error-level,
this accuracy compromise is relevant to time step in observer computation
- Other blade conditions present reliable accuracy.
'''
import numpy as np
import csdl_alpha as csdl
from lsdo_acoustics.core.acoustics import Acoustics
from revised.BPM_model import BPMVariableGroup, BPM_model
from revised.A_weighting_function import A_weighting_function

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


# ==================== Blade condition : 'S76' ========================
recorder = csdl.Recorder(inline = True)
recorder.start()

flight_condition = 'hover'
outband = 'one third'

x = 2.1075
y = 0
z = 0.3716
num_observers = 1

obs_x = csdl.Variable(value = x)  #csdl.var

observer_data = {'x': obs_x,
                 'y': y,
                 'z': z,
                 'num_observers': num_observers
                 }


num_blades = 4
RPM = 1670
radius = 1.07

M = 0
rho = 1.225 # air density
mu = 1.789*(1e-5) # dynamic_viscosity
c0 = 340.3  # sound of speed

freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                 6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]
num_freq = len(freq)

chord = 0.0630
radial = np.linspace(0.225, 1.059, 40)
sectional_span = radial[2] - radial[1]
num_radial = len(radial)
azimuth = np.linspace(0, 2*np.pi, 40)
num_azim = len(azimuth)
num_nodes = 1 ## 

AOA = np.array([-1.7160, -0.4130, 0.6480, 1.5150, 2.2270,
                 2.8120,  3.2920, 3.6860, 4.0080, 4.2710,
                 4.4850,  4.6570, 4.7950, 4.9050, 4.9920,
                 5.0600,  5.1130, 5.1550, 5.1880, 5.2150,
                 5.2380,  5.2600, 5.2820, 5.3060, 5.3340,
                 5.3670,  5.4060, 5.4530, 5.5090, 5.5740,
                 5.6510,  5.7390, 5.8390, 5.9540, 6.0820, 
                 6.2250,  6.3840, 6.5590, 6.7500, 6.9590])
                
A_cor = 0.
a_star = AOA + A_cor

# Velocity computation
RPM =csdl.Variable(value = RPM)   # csdl.var
omega = RPM*2.*np.pi/60.
Vinf = 0    # In hover condition, Vinf = 0
V0 = Vinf + omega*radius
                
V0_init = np.array([42.9780,  46.5300,  50.1180,  53.7340,  57.3740, 
                    61.0320,  64.7050,  68.3910,  72.0860,  75.7880,
                    79.4970,  83.2110,  86.9280,  90.6470,  94.3680, 
                    98.0890,  101.8090, 105.5270, 109.2440, 112.9570,
                    116.6670, 120.3730, 124.0740, 127.7700, 131.4600, 
                    135.1440, 138.8220, 142.4940, 146.1590, 149.8170,
                    153.4680, 157.1130, 160.7520, 164.3840, 168.0110,
                    171.6340, 175.2530, 178.8690, 182.4830, 186.0970])


pitch = np.array([7.89719626168224, 7.70093457943925, 7.49532710280374, 7.29906542056075,
                  7.10280373831776, 6.89719626168224, 6.70093457943925, 6.49532710280374, 
                  6.29906542056075, 6.10280373831776, 5.89719626168224, 5.70093457943925,	
                  5.49532710280374, 5.29906542056075, 5.10280373831776, 4.89719626168224,
                  4.70093457943925, 4.49532710280374, 4.29906542056075, 4.10280373831776,
                  3.89719626168224, 3.70093457943925, 3.49532710280374, 3.29906542056075,
                  3.10280373831776, 2.89719626168224, 2.70093457943925, 2.49532710280374,
                  2.29906542056075, 2.10280373831776, 1.89719626168224, 1.70093457943925,	
                  1.49532710280374, 1.29906542056075, 1.10280373831776, 0.89719626168224,	
                  0.70093457943925, 0.49532710280374, 0.29906542056075, 0.10280373831776])

alpha = 0
sigma = 0.075
TE_thick = 4.3e-4 #h
slope_angle = 19.  #Psi

# ============================= SPL computation =============================
BPM_vg = BPMVariableGroup(
    chord=chord,
    radial=radial, #R
    sectional_span=sectional_span, # R(1) - R(2) / l
    a_star=a_star, # AoAcor = 0. / a_star = AOA - AOAcor.
    pitch=pitch,
    azimuth=azimuth,
    alpha=alpha,
    TE_thick=TE_thick, #h
    slope_angle=slope_angle, #Psi
    free_vel=V0, #free-stream velocity U and V0
    freq=freq,
    RPM=RPM,
    speed_of_sound=c0, #c0
    Vinf = Vinf,
    density=rho,  #rho
    dynamic_viscosity=mu,  #mu
    num_radial=num_radial,
    num_tangential=num_azim,   # num_tangential = num_azim
    num_freq=num_freq
    )
                          
SPL_rotor, OASPL = BPM_model(BPMVariableGroup = BPM_vg,
                             observer_data = observer_data,
                             num_blades = num_blades,
                             num_nodes = 1,
                             flight_condition = flight_condition,
                             outband = outband
                             )

print('rotor SPL :', SPL_rotor.value)
print('OASPL :', OASPL.value)

# =========================== A-weighting computation ===========================
SPL_rotor_A = A_weighting_function(SPL_rotor, freq)
print('A-weighting:', SPL_rotor_A.value)

# =========================== Derivative computation ============================
asdf = csdl.derivative(ofs=OASPL, wrts=RPM)
print(f'derivative value: {asdf.value}')
asdf = csdl.derivative_utils.verify_derivatives(ofs=OASPL, wrts=RPM, step_size=1.e-6)

# asdf1 = csdl.derivative(ofs=OASPL, wrts=slope_angle)
# print(f'derivative value: {asdf1.value}')
# asdf1 = csdl.derivative_utils.verify_derivatives(ofs=OASPL, wrts=slope_angle, step_size=1.e-6)

# # =================== HJ's ref. data for Verification =========================
import matplotlib.pyplot as plt

BPM_HJ_SPL_rotor = np.array([59.76355553, 61.70342095, 63.75511851, 66.30118566,
                             69.22452662, 72.32857047, 75.55801059, 78.22070099,
                             80.65988131, 81.36054253, 79.66754845, 76.82268446,
                             73.86261434, 70.77617849, 68.39617445, 67.09474199,
                             66.49159409, 66.27263448, 66.29427156, 65.16905762,
                             57.73327168, 55.28776975])


BPM_HJ_OASPL = 87.7548

SPL_rotor = SPL_rotor.value.reshape(-1)
rel_error = abs((BPM_HJ_SPL_rotor - SPL_rotor))/BPM_HJ_SPL_rotor
OASPL_rel_error = abs((BPM_HJ_OASPL - OASPL).value/ BPM_HJ_OASPL)
print('Relative error of BPM with respect to Frequency :', rel_error)

# # SPL rotor realtive error plot
# plt.figure()
# plt.semilogy(freq, rel_error)
# # plt.semilogy(freq, rel_error2, label = 'V0 : RPM*2pi/60*radius')
# # plt.legend(['V0 : from HJ data', 'V0 : RPM*2pi/60*radius'])
# plt.title('Relative error of BPM model')
# plt.xlabel('Frequency [HZ]')
# plt.ylabel('Relative error')
# # plt.ylim([1e-4, 6e-3])
# plt.grid()
# plt.show()

# SPL rotor A-weighting value plo t
SPL_rotor_A = SPL_rotor_A.value.reshape(-1)
plt.figure()
plt.plot(freq, SPL_rotor)
plt.plot(freq, SPL_rotor_A)
plt.legend(['SPL [dB]', 'A-weighting [dBA]'])
plt.title('Rotor SPL and A-weighting values')
plt.xlabel('Frequency [HZ]')
plt.ylabel('SPL values')
plt.show()
import numpy as np
import csdl_alpha as csdl
from lsdo_acoustics.core.acoustics import Acoustics
from revised.BPM_model import BPMVariableGroup, BPM_model
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

M = 0
rho = 1.225 # air density
mu = 1.789*(1e-5) # dynamic_viscosity
c0 = 340.3  # sound of speed
omega = RPM*2.*np.pi/60.
Vinf = M*c0

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
azimuth = np.linspace(0, 2*np.pi, 20)   # need to be checked, if single colmn azimuthal does not work, modification is needed.
num_azim = len(azimuth)
num_nodes = 1 ## 

AOA = edgewise_input['AoA']
# AOA = np.array([3.1870,	4.8630,	5.5070, 5.7190, 5.7150,	
#                 5.6030,	5.4370, 5.2490, 5.0550, 4.8640,
#                 4.6790, 4.5030, 4.3370, 4.1810, 4.0340,
#                 3.8960, 3.7670, 3.6460, 3.5320, 3.4240, 
#                 3.3230, 3.2270, 3.1360, 3.0510, 2.9690, 
#                 2.8920, 2.8180, 2.7480, 2.6800, 2.6150,	 
#                 2.5500, 2.4870, 2.4220, 2.3540,	 2.2780,	
#                 2.1900, 2.0760, 1.9180, 1.6670, 1.1480])
                
A_cor = 0.
a_star = AOA + A_cor
   
V0 = edgewise_input['V0']     
V0 = (V0**2)**0.5        
# V0 = np.array([20.5490, 22.2650, 24.0070, 25.7630, 27.5310,
#                29.3070, 31.0900, 32.8770, 34.6690, 36.4640,
#                38.2610, 40.0610, 41.8630, 43.6660, 45.4720,
#                47.2790, 49.0870, 50.8970, 52.7070, 54.5190,
#                56.3320, 58.1450, 59.9600, 61.7750, 63.5910,
#                65.4070, 67.2250, 69.0430, 70.8610, 72.6800,
#                74.5000, 76.3200, 78.1410, 79.9640, 81.7880,
#                83.6140, 85.4450, 87.2820, 89.1330, 91.0260])


pitch = np.array([5.71827196, 9.73526282, 13.81025512, 17.0172273,  18.47316765,
                  18.0755777, 17.1715650, 16.19259423, 15.24232188, 14.33239574,
                  13.4617839, 12.5032064, 11.8446102,  11.18607406, 10.5113536 ,
                  9.88999153,  9.3711866,  8.89386211,  8.57064106,  8.19189033,
                  7.76509555,  7.3859008,  7.06844137,  6.70339299,  6.45105441,
                  6.24791936,  6.0390358,  5.79490773,  5.57605766,  5.36631695,
                  5.17705074,  4.9779330,  4.83475723,  4.74343221,  4.66031611,
                  4.54590395,  4.4733133,  4.42699749,  4.4035366 ,  4.39935211])

alpha = 0
sigma = 0.0712
# tau = 30*azimuth/(np.pi*RPM)   # source time
TE_thick = 7.62e-4 # h
slope_angle = 19.  # Psi

recorder = csdl.Recorder(inline = True)
recorder.start()

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
           
TBLTE_dep, spl_BLUNT, spl_LBLVS, observer_data_tr = BPM_model(BPMVariableGroup = BPM_vg,
                                                              observer_data = observer_data,
                                                              num_observers = observer_data['num_observers'],
                                                              num_blades = num_blades,
                                                              num_nodes = 1,
                                                              flight_condition = flight_condition
                                                              )
splp = TBLTE_dep['splp']
spls = TBLTE_dep['spls']
spla = TBLTE_dep['spla']

BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) # + csdl.power(10, spl_LBLVS) : untripped condition
totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20
totalSPL0 = totalSPL  # just for check before outband transformation

# outband transformation
if outband == 'one third':
    totalSPL = totalSPL
else:
    target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_freq)
    exp_freq = csdl.expand(freq, target_shape, 'i->abcdei')
    narrowSPL = totalSPL - 10.*csdl.log(0.2315*exp_freq, 10)
    totalSPL = 10.*csdl.log(int(outband)*csdl.power(10, narrowSPL/10), 10)
    
# ================= Find time index according to time frame ===================
obs_time = observer_data_tr['obs_time']
tMin = csdl.minimum(obs_time, rho = 1000000.).value
tMax = csdl.maximum(obs_time, rho = 1000000.).value

num_tRange = 40  # # of time step is arbitrary chosen!! 
tRange = np.linspace(tMin, tMax, num_tRange)
dt = tRange[2] - tRange[1]

time_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_tRange)
exp_tRange = csdl.expand(csdl.reshape(tRange, (num_tRange,)), time_shape, 'i->abcdei')
exp2_obs_time = csdl.expand(obs_time, time_shape, 'ijklm->ijklma')

time_dist = ((exp_tRange - exp2_obs_time)**2)**0.5
arr_time_dist = time_dist.value

sorted_dist = np.argsort(arr_time_dist, axis = -1)  
time_indx1 = sorted_dist[:, :, :, :, :, 0]    # smallest dist index (Idx) 
time_indx2 = sorted_dist[:, :, :, :, :, 1]    # 2nd smallest dist index (Idx+1 or Idx-1)

min_tRange1  = np.take(exp_tRange.value, time_indx1) # Select appropriate time value to be allocated
min_tRange2 = np.take(exp_tRange.value, time_indx2)  # Select appropriate time value 2 to be allocated

target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_freq)
time_coeff1 = csdl.expand((((min_tRange1 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')
time_coeff2 = csdl.expand((((min_tRange2 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')

# Compute noise contribution
noise_con1 = time_coeff1*csdl.power(10, totalSPL/10)
noise_con2 = time_coeff2*csdl.power(10, totalSPL/10)

# ================ Allocate noise contribution to time frame ==================
start = time.time()
time_indx1 = csdl.Variable(value = time_indx1)
time_indx2 = csdl.Variable(value = time_indx2)
sumSPL = csdl.Variable(shape=(num_nodes, num_observers, num_tRange, num_freq), value=0)
for k in csdl.frange(num_blades):
    for j in csdl.frange(num_azim): 
        for i in csdl.frange(num_radial):
            closestIndx = time_indx1[:, :, i, j, k]
            closestIndx2 = time_indx2[:, :, i, j, k]
            sumSPL = sumSPL.set(csdl.slice[:, :, closestIndx, :], sumSPL[:, :, closestIndx, :] + noise_con1[:, :, i, j, k, :])
            sumSPL = sumSPL.set(csdl.slice[:, :, closestIndx2, :], sumSPL[:, :, closestIndx2, :] + noise_con2[:, :, i, j, k, :])
end = time.time()
print('time consuming for HJ loops :', end - start)

rotor_SPL = 10*csdl.log(sumSPL, 10)
final_SPL = csdl.sum(rotor_SPL, axes=(2,))/num_tRange
print('final SPL is :', final_SPL.value)
A = final_SPL[0, 0, :].value

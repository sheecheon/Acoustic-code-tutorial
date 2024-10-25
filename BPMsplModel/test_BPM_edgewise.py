import numpy as np
import csdl_alpha as csdl
from lsdo_acoustics.core.acoustics import Acoustics
from revised.BPM_model import BPMVariableGroup, BPM_model
import time
import pickle
from scipy import io

recorder = csdl.Recorder(inline = True)
recorder.start()
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

with open('edgewise_input_tensor', 'rb') as f:
    edgewise_input = pickle.load(f)
    
# ======================= Blade condition : 'Burley' ==========================
flight_condition = 'edgewise'
outband = '70'
x = -2.
y = 1.6
z = -2.3
num_observers = 1
observer_data = {'x': x,
                 'y': y,
                 'z': z,
                 'num_observers': num_observers
                 }

num_blades = 4
RPM = 1040
radius = 2

M = 0.0960
rho = 1.225 # air density
mu = 1.789*(1e-5) # dynamic_viscosity
c0 = 340.3  # sound of speed
omega = RPM*2.*np.pi/60.
Vinf = M*c0

freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                 6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]
num_freq = len(freq)

chord = 0.1210
radial = np.linspace(0.42, 1.98, 40)
sectional_span = radial[2] - radial[1]
num_radial = len(radial)
azimuth = np.linspace(0, 2*np.pi, 40)   # need to be checked, if single colmn azimuthal does not work, modification is needed.
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
   
RPM =csdl.Variable(value = RPM)   # csdl.var
omega = RPM*2.*np.pi/60.
azim_exp = csdl.expand(azimuth, (num_azim, num_radial, num_blades),'i->iab')
Vr = omega*radial
Vr_exp = csdl.expand(Vr, (num_azim, num_radial, num_blades),'i->aib')
azim_exp = azim_exp*np.pi/180
V0 = Vr_exp + Vinf*csdl.sin(azim_exp)

pitch = np.array([10.3200, 10.1600, 10.0000, 9.8400, 9.6800, 
                    9.5200, 9.3600, 9.2000, 9.0400, 8.8800,
                    8.7200, 8.5600, 8.4000, 8.2400, 8.0800, 
                    7.9200, 7.7600, 7.6000, 7.4400, 7.2800,
                    7.1200, 6.9600, 6.8000, 6.6400, 6.4800, 
                    6.3200, 6.1600, 6.0000, 5.8400, 5.6800,
                    5.5200, 5.3600, 5.2000, 5.0400, 4.8800, 
                    4.7200, 4.5600, 4.4000, 4.2400, 4.0800]) 


alpha = -5.3
sigma = 0.077
# tau = 30*azimuth/(np.pi*RPM)   # source time
TE_thick = 6.e-04  # h
slope_angle = 10.7100  # Psi


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

time_coeff1 = csdl.expand((((min_tRange1 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')
time_coeff2 = csdl.expand((((min_tRange2 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')

# Compute noise contribution
noise_con1 = time_coeff1*csdl.power(10, totalSPL/10)
noise_con2 = time_coeff2*csdl.power(10, totalSPL/10)

# ================ Allocate noise contribution to time frame ==================
start0 = time.time()
time_indx01 = csdl.Variable(value = time_indx1)
time_indx02 = csdl.Variable(value = time_indx2)
sumSPL0 = csdl.Variable(shape=(num_nodes, num_observers, num_tRange, num_freq), value=0)
for k in csdl.frange(num_blades):
    for j in csdl.frange(num_radial):
        for i in csdl.frange(num_azim):
            closestIndx = time_indx01[:, :, i, j, k]
            closestIndx2 = time_indx02[:, :, i, j, k]
            sumSPL0 = sumSPL0.set(csdl.slice[:, :, closestIndx, :], sumSPL0[:, :, closestIndx, :] + noise_con1[:, :, j, i, k, :])
            sumSPL0 = sumSPL0.set(csdl.slice[:, :, closestIndx2, :], sumSPL0[:, :, closestIndx2, :] + noise_con2[:, :, j, i, k, :])
end0 = time.time()
print('time consuming for HJ loops :', end0 - start0)

rotor_SPL0 = 10*csdl.log(sumSPL0, 10)
final_SPL0 = csdl.sum(rotor_SPL0, axes=(2,))/num_tRange
print('final SPL is :', final_SPL0.value)
# A = final_SPL0.value

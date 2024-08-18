import numpy as np
import csdl_alpha as csdl 
import pickle
from scipy import io
import time
from BPM_spl_edge import BPMsplModel

recorder = csdl.Recorder(inline = True)
recorder.start()

# def obs_time_computation():
    
#     return obs_time

#load input for Edgewise condition from HJ      
with open('EdgewiseInput_SUI', 'rb') as f:                             
# with open('EdgewiseInput_Burley', 'rb') as f:
    Edgewise_input = pickle.load(f)
    
class DummyMesh(object):
    def __init__(self, num_radial, num_tangential):
        self.parameters = {
            'num_radial': num_radial,
            'num_tangential': num_tangential,
            'mesh_units': 'm'
        }

radial = Edgewise_input['R']
azimuth = Edgewise_input['azimuth']
pitch = Edgewise_input['pitch']
# alpha = Edgewise_input['alpha']
freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,
                  8000,10000,12500,16000,20000,25000,31500,40000,50000]) # 1/3 octave band central frequency [Hz]
# freq = np.array([400])

rpm = Edgewise_input['RPM']
alpha = Edgewise_input['alpha']
Psi = Edgewise_input['Psi']

r = Edgewise_input['R']       #sectional radius
M = Edgewise_input['M']
# U = M*c0

num_nodes = 1
num_radial = radial.size # should be len(radial) (0.022 ~ 0.14[m] by 40 sections)
num_tangential = len(azimuth)
num_azim = num_tangential
num_observers = 1
num_freq = len(freq)   
num_blades = Edgewise_input['B']
                
mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

# observer time computation
alpha = alpha*(2*np.pi)/180
tau = 30*azimuth[:,0]/(np.pi*rpm)
# obs_t0 = 30*Psi/(np.pi*4047)   # obs_t0 = 30*Psi/(np.pi*rpm)   # initial guess for Newton method
# obs_t = csdl.ImplicitVariable(name='obs_t', value = obs_t0.value)   #0.04483238

# OBS time convert : should be "obs_time = obs_time()" / Current: import from HJ's code due to memory issue.
obs_time = io.loadmat('OBStime_SUI.mat')
# obs_time = io.loadmat('OBStime_Burley.mat')
obs_time = obs_time['ObsTime']     #s hape = (num_azim, num_radial , num_blades)

tMin = csdl.minimum(obs_time, rho = 1000000.).value
tMax = csdl.maximum(obs_time, rho = 1000000.).value

# Distance btw observer to blade section ref. from HJ
obs_x = Edgewise_input['X']
obs_y = Edgewise_input['Y']
obs_z = Edgewise_input['Z']

# target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq, num_blades)
target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_freq)

exp_x = csdl.expand(obs_x, target_shape)
exp_y = csdl.expand(obs_y, target_shape)
exp_z = csdl.expand(obs_z, target_shape)

# exp_r = csdl.expand(radial, target_shape, 'ij->aijbcde')
exp_r = csdl.expand(csdl.reshape(radial, (num_radial,)), target_shape, 'i->abicde')
exp_azimuth = csdl.expand(azimuth, target_shape, 'ij->abcijd')
exp_alpha = csdl.expand(alpha, target_shape)    #the size of 'alpha' needs to be checked.
exp_tau = csdl.expand(tau, target_shape, 'i->abcide')

exp_obs_time = csdl.expand(obs_time, target_shape, 'ijk->abjikc')

Vinf = Edgewise_input['M']*Edgewise_input['a']

obs_position_tr = {'x_r' : exp_x - exp_r*csdl.cos(exp_azimuth)*csdl.cos(alpha) + Vinf*(exp_obs_time - exp_tau),
                   'y_r' : exp_y - exp_r*csdl.sin(exp_azimuth)*csdl.cos(alpha),
                   'z_r' : exp_z - exp_r*csdl.cos(exp_azimuth)*csdl.sin(alpha)
                   }

# ========== Computation total BPM spl with respect to 4 sub-models ===========
BPM = BPMsplModel(Edgewise_input, obs_position_tr, num_observers, num_radial, num_tangential, num_freq, num_nodes = 1)

splp, spls, spla, spl_TBLTE, spl_TBLTE_cor = BPM.TBLTE()
# splp_val = splp[0, 0, -1, -1, -1, :].value
# spls_val = spls[0, 0, -1, -1, -1, :].value
# spla_val = spla[0, 0, -1, -1, -1, :].value
# splTBLTE_val = spl_TBLTE[0, 0, -1, -1, -1, :].value

spl_BLUNT = BPM.TE_BLUNT()
# splBLUNT_val = spl_BLUNT[0, 0, -1, -1, -1, :].value

spl_LBLVS = BPM.LBLVS()
# splLBLVs_val = spl_LBLVS[0, 0, -1, -1, -1, :].value

BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) #+ csdl.power(10, spl_LBLVS) : untripped condition
totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20 / SPL13 in ref. HJ's code
# A = totalSPL[0, 0, -1, -1, -1, :].value

# ==================== Find closest & 2nd closet time index ===================
num_tRange = 40  # # of time step is arbitrary chosen!! 
tRange = np.linspace(tMin, tMax, num_tRange)
dt = tRange[2] - tRange[1]

time_shape = (num_azim, num_radial, num_blades, num_tRange)
exp_tRange = csdl.expand(csdl.reshape(tRange, (num_tRange,)), time_shape, 'i->abci')
exp2_obs_time = csdl.expand(obs_time, time_shape, 'ijk->ijka')

time_dist = ((exp_tRange - exp2_obs_time)**2)**0.5
arr_time_dist = time_dist.value

sorted_dist = np.argsort(arr_time_dist, axis = -1)  # dim = (40, 40, 4, 35)
time_indx1 = sorted_dist[:, :, :, 0]    # smallest dist index (Idx) / dim = (40, 40, 4)
time_indx2 = sorted_dist[:, :, :, 1]   # 2nd smallest dist index (Idx+1 or Idx-1) / dim = (40, 40, 4)

min_tRange1  = np.take(exp_tRange.value, time_indx1) # Select appropriate time value to be allocated / dim = (40, 40, 4)
min_tRange2 = np.take(exp_tRange.value, time_indx2)  # Select appropriate time value 2 to be allocated / dim = (40, 40, 4)

time_coeff1 = csdl.expand((((min_tRange1 - obs_time)**2)**0.5)/dt, target_shape, 'ijk->abjikc')
time_coeff2 = csdl.expand((((min_tRange2 - obs_time)**2)**0.5)/dt, target_shape, 'ijk->abjikc')

# Compute noise contribution
noise_con1 = time_coeff1*csdl.power(10, totalSPL/10)
noise_con2 = time_coeff2*csdl.power(10, totalSPL/10)

# ============ Try1 & Try 2 : reshape for noise contribution allocation ==============
re_shape = num_nodes*num_observers*num_radial*num_azim*num_blades
re_noise_con1 = csdl.reshape(csdl.reorder_axes(noise_con1, 'ijklmn->ijlkmn'), (re_shape, num_freq))
re_noise_con2 = csdl.reshape(csdl.reorder_axes(noise_con2, 'ijklmn->ijlkmn'), (re_shape, num_freq))
re_time_indx1  = csdl.reshape(time_indx1, (re_shape,))
re_time_indx2 = csdl.reshape(time_indx2, (re_shape,))

allocate_shape = (num_nodes*num_observers*num_radial*num_azim*num_blades, num_tRange, num_freq)
sumSPL1 = csdl.Variable(shape = allocate_shape, value = 0.)   # empty tensor
sumSPL2 = csdl.Variable(shape = allocate_shape, value = 0.)   # empty tensor
sumSPL = csdl.Variable(shape = allocate_shape, value = 0.)   # empty tensor

# Try1 : without 'for' loops
# sumSPL1 = sumSPL1.set(csdl.slice[0:, re_time_indx1[:], :], re_noise_con1[0:,:])
# sumSPL2 = sumSPL1.set(csdl.slice[0:, re_time_indx2[:], :], re_noise_con2[0:,:])
# sumSPL = sumSPL1 + sumSPL2
# total_SPL = csdl.sum(sumSPL, axes=(0,))
# # A = total_SPL.value

# rotor_SPL = 10*csdl.log(total_SPL, 10)
# final_SPL = csdl.sum(rotor_SPL, axes=(0,))/num_tRange
# final_SPL = np.nan_to_num(final_SPL)
# print(final_SPL.value)


# Try2 : with 'for' loops
start = time.time()
for i in csdl.frange(re_shape):
    sumSPL = sumSPL.set(csdl.slice[i, re_time_indx1[i], :], sumSPL[i, re_time_indx1[i], :] + re_noise_con1[i, :])   # 12.35s
    sumSPL = sumSPL.set(csdl.slice[i, re_time_indx2[i], :], sumSPL[i, re_time_indx2[i], :] + re_noise_con2[i, :])
    # sumSPL1 = sumSPL1.set(csdl.slice[i, re_time_indx1[i], :], re_noise_con1[i, :])   # 24.54s
    # sumSPL2 = sumSPL2.set(csdl.slice[i, re_time_indx2[i], :], re_noise_con2[i, :])
end = time.time()
print('time consuming for loops :', end - start)
# sumSPL = sumSPL1 + sumSPL2
total_SPL = csdl.sum(sumSPL, axes=(0,))

rotor_SPL = 10*csdl.log(total_SPL, 10)
final_SPL = csdl.sum(rotor_SPL, axes=(0,))/num_tRange
final_SPL = np.nan_to_num(final_SPL)
print(final_SPL.value)

# ============== Following HJ's code (0.082s) for verification ==================
# start0 = time.time()
# time_indx01 = csdl.Variable(value = time_indx1)
# time_indx02 = csdl.Variable(value = time_indx2)
# sumSPL0 = csdl.Variable(shape=(num_tRange, num_freq), value=0)
# for k in csdl.frange(num_blades):
#     for j in csdl.frange(num_radial):
#         for i in csdl.frange(num_azim):
#             closestIndx = time_indx01[i, j, k]
#             closestIndx2 = time_indx02[i, j, k]
#             sumSPL0 = sumSPL0.set(csdl.slice[closestIndx, :], sumSPL0[closestIndx, :] + noise_con1[0, 0, j, i, k, :])
#             sumSPL0 =  sumSPL0.set(csdl.slice[closestIndx2, :], sumSPL0[closestIndx2, :] + noise_con2[0, 0, j, i, k, :])
# end0 = time.time()
# print('time consuming for HJ loops :', end0 - start0)
# 
# rotor_SPL0 = 10*csdl.log(sumSPL0, 10)
# final_SPL0 = csdl.sum(rotor_SPL0, axes=(0,))/num_tRange
# print(final_SPL0.value)


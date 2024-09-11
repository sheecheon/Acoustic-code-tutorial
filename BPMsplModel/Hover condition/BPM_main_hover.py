import numpy as np
import csdl_alpha as csdl 
import pickle
from BPM_spl_model import BPMsplModel

recorder = csdl.Recorder(inline = True)
recorder.start()

# Coordinate transformation for 'Hover' condition
def compute_rotor_frame_position(sectional_x, sectional_y, sectional_z, pitch, azimuth, num_observers, num_radial, num_azim, num_freq, num_nodes =1):
    
    target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq)
    twist_exp = csdl.expand(csdl.reshape(pitch*(np.pi/180),(num_radial,)), target_shape, 'i->abicd')
    azim_exp = csdl.expand(azimuth, target_shape, 'i->abcid')
    
    
    beta_p = 0.   # flapping angle
    coll = 0.     # collective pitch
    
    sin_th = csdl.sin(twist_exp)
    cos_th = csdl.cos(twist_exp)
    sin_ph = csdl.sin(azim_exp)
    cos_ph = csdl.cos(azim_exp)
    # According to ref. HJ, M_beta = 0, last term does not account for this file
     
    x_r = sectional_x*cos_th*sin_ph - sectional_y*cos_ph*cos_th + sectional_z*sin_th
    y_r = sectional_x*cos_ph + sectional_y*sin_ph
    z_r = -sectional_x*sin_th*sin_ph + sectional_y*cos_ph*sin_th + sectional_z*cos_th
    
    obs_position_tr = {
        'x_r': x_r,
        'y_r': y_r,
        'z_r': z_r
        }
    
    return obs_position_tr

#load input for Hover condition from HJ   
# with open('HoverInput_S76', 'rb') as f:                            
# with open('HoverInput_APC', 'rb') as f:
with open('HoverInput_Idealtwist', 'rb') as f:
    HoverInput = pickle.load(f)
    
class DummyMesh(object):
    def __init__(self, num_radial, num_tangential):
        self.parameters = {
            'num_radial': num_radial,
            'num_tangential': num_tangential,
            'mesh_units': 'm'
        }

radial = HoverInput['R']  #[0, 0:10]
azimuth = HoverInput['azimuth'][:,0]
# azimuth = HoverInput['azimuth']
pitch = HoverInput['pitch']
freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                  6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]


num_nodes = 1
num_radial = radial.size # should be len(radial) (0.022 ~ 0.14[m] by 40 sections)
num_tangential = len(azimuth)
num_azim = num_tangential
num_observers = 1
num_freq = len(freq)   
num_blades = HoverInput['B']
                
mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

# Distance btw observer to blade section ref. from HJ
obs_x = HoverInput['X']
obs_y = HoverInput['Y']
obs_z = HoverInput['Z']

target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq) #num_tangential = num_azimuth
exp_x = csdl.expand(obs_x, target_shape)
exp_y = csdl.expand(obs_y, target_shape)
exp_z = csdl.expand(obs_z, target_shape)
exp_azimuth = csdl.expand(azimuth, target_shape, 'i->abcid')
exp_radial = csdl.expand(csdl.reshape(radial, (num_radial,)), target_shape, 'i->abicd')

xloc = exp_radial*csdl.cos(exp_azimuth)
yloc = exp_radial*csdl.sin(exp_azimuth)

sectional_x = exp_x - xloc    #newX
sectional_y = exp_y - yloc    #newY
sectional_z = exp_z
obs_position_tr = compute_rotor_frame_position(sectional_x, sectional_y, sectional_z, pitch, azimuth, num_observers, num_radial, num_azim, num_freq)
                                                    
# Computation total BPM spl with 4 subcomponents
BPM = BPMsplModel(HoverInput, obs_position_tr, num_observers, num_radial, num_tangential, num_freq, num_nodes = 1)

splp, spls, spla, spl_TBLTE, spl_TBLTE_cor = BPM.TBLTE()
splp_val = splp[0, 0, -1, -1, :].value
spls_val = spls[0, 0, -1, -1, :].value
spla_val = spla[0, 0, -1, -1, :].value
splTBLTE_val = spl_TBLTE[0, 0, -1, -1, :].value

spl_BLUNT = BPM.TE_BLUNT()
splBLUNT_val = spl_BLUNT[0, 0, -1, -1, :].value

spl_LBLVS = BPM.LBLVS()
splLBLVs_val = spl_LBLVS[0, 0, -1, -1, :].value

BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) # csdl.power(10, spl_LBLVS) : untripped condition
totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20

# csdl.power check!! - spl_BLUNT : 16th value (APC hover)
# csdl.power(10, 24.10998103/10).value   #ans: 257.63099
# csdl.power(10, 30.1771/10).value       #ans: 1041.6216

# ======================= Final Rotor spl computation =========================
x_tr = obs_position_tr['x_r']
y_tr = obs_position_tr['y_r']
z_tr = obs_position_tr['z_r']

S_tr = ((x_tr**2) + (y_tr**2) + (z_tr**2))**0.5

U = HoverInput['V0'][0,:,0]
# U = HoverInput['V0']
c0 = HoverInput['a']
# Mr = csdl.expand(U/c0, target_shape, 'ij->aijbc')
Mr = csdl.expand(U/c0, target_shape, 'i->abicd')   
W = 1 + Mr*(x_tr/S_tr)

# =================== Intial computation for rotor SPL ========================
Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)
Spp_func = (2*np.pi/(num_azim-1))*(W**2)*Spp_bar      # Spp_func = (W**2)*Spp_bar  # ok
Spp_R = num_blades*(1/(2*np.pi))*csdl.sum(Spp_func, axes=(3,))
Spp_rotor = csdl.sum(Spp_R, axes=(2,))

SPL_rotor = 10*csdl.log(Spp_rotor, 10)
OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

print('OASPL : ', OASPL.value)
A = SPL_rotor[0, 0, :].value

# # ======================== Verification w. HJ's code ==========================
# Spp_bar_TBLTE_SS = csdl.power(10, spls/10) 
# Spp_bar_TBLTE_a = csdl.power(10, spla/10) 

# Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)
# func = (W**2)*Spp_bar
# func_TBLTE_SS = (W**2)*Spp_bar_TBLTE_SS
# func_TBLTE_a = (W**2)*Spp_bar_TBLTE_a


# # A = func[0, 0, -1, :, :].value
# Spp_TBLTE_SS = (1/(2*np.pi))*((2*np.pi)/(num_azim-1))*csdl.sum(func_TBLTE_SS, axes=(3,))
# Spp_TBLTE_a = (1/(2*np.pi))*((2*np.pi)/(num_azim-1))*csdl.sum(func_TBLTE_a, axes=(3,))

# Spp_func = ((2*np.pi)/(num_azim-1))*csdl.sum(func, axes=(3,))
# Spp = (1/(2*np.pi))*Spp_func

# azimuthalSpp = num_blades*Spp

# totalSpp = csdl.sum(azimuthalSpp, axes=(2,))

# SPL_rotor = 10*csdl.log(totalSpp, 10)
# OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

# print('OASPL : ', OASPL.value)
# A = SPL_rotor[0, 0, :].value

# ======================== Integration test :trapz ============================
# Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)
# integ_func = (W**2)*Spp_bar
# # # case : index1
# # integ_func = integ_func.set(csdl.slice[0, 0, :, 0, :], value = 0.5*integ_func[0, 0, :, 0, :])  
# # integ_func = integ_func.set(csdl.slice[0, 0, :, -1, :], value = 0.5*integ_func[0, 0, :, -1, :])

# # case : index2
# # integ_func = integ_func.set(csdl.slice[0, 0, :, :, 0], value = 0.5*integ_func[0, 0, :, :, 0])
# # integ_func = integ_func.set(csdl.slice[0, 0, :, :, -1], value = 0.5*integ_func[0, 0, :, :, -1])

# # case : index3 - right side Riemann sum
# # integ_func = integ_func.set(csdl.slice[0, 0, :, 0, :], value = 0.)

# Spp_func =(2*np.pi/(num_azim-1))*csdl.sum(integ_func, axes=(3,))
# Spp = (1/(2*np.pi))*Spp_func

# Spp_rotor = csdl.sum(Spp, axes=(2,))

# SPL_rotor = 10*csdl.log(Spp_rotor, 10)
# OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

# print('OASPL : ', OASPL.value)
# A = SPL_rotor[0, 0, :].value 


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
with open('Hover_input', 'rb') as f:
    HoverInput = pickle.load(f)
    
class DummyMesh(object):
    def __init__(self, num_radial, num_tangential):
        self.parameters = {
            'num_radial': num_radial,
            'num_tangential': num_tangential,
            'mesh_units': 'm'
        }

radial = HoverInput['R']
azimuth = HoverInput['azimuth']
pitch = HoverInput['pitch']
freq = HoverInput['f']

num_nodes = 1
num_radial = len(azimuth) # should be len(radial) (0.022 ~ 0.14[m] by 40 sections)
num_tangential = len(azimuth)
num_azim = num_tangential
num_observers = len(HoverInput['X'])
num_freq = len(freq)   
num_blades = HoverInput['B'].astype('int')
# num_blades = HoverInput['B']
                
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
                                                    
# Computation total BPM spl with respect to 4 sub-models
BPM = BPMsplModel(HoverInput, obs_position_tr, num_observers, num_radial, num_tangential, num_freq, num_nodes = 1)

splp, spls, spla, spl_TBLTE, spl_TBLTE_cor = BPM.TBLTE()
# splp_val = splp[0, 0, -1, -1, :].value
# spls_val = spls[0, 0, -1, -1, :].value
# spla_val = spla[0, 0, -1, -1, :].value
# splTBLTE_val = spl_TBLTE[0, 0, -1, -1, :].value

spl_BLUNT = BPM.TE_BLUNT()
# splBLUNT_val = spl_BLUNT[0, 0, -1, -1, :].value

spl_LBLVS = BPM.LBLVS()
# splLBLVs_val = spl_LBLVS[0, 0, -1, -1, :].value

BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) #+ csdl.power(10, spl_LBLVS) : untripped condition
totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20
Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)

# ======================= Final Rotor spl computation =========================
x_tr = obs_position_tr['x_r']
y_tr = obs_position_tr['y_r']
z_tr = obs_position_tr['z_r']

S_tr = ((x_tr**2) + (y_tr**2) + (z_tr**2))**0.5

U = HoverInput['V0']
c0 = HoverInput['a']
Mr = csdl.expand(U/c0, target_shape, 'ij->aijbc')
W = 1 + Mr*(x_tr/S_tr)

Spp_func = (2*np.pi/num_azim)*(W**2)*Spp_bar      # Spp_func = (W**2)*Spp_bar  # ok
Spp_R = num_blades*(1/2*np.pi)*csdl.sum(Spp_func, axes=(3,))

Spp_rotor = csdl.sum(Spp_R, axes=(2,))

SPL_rotor = 10*csdl.log(Spp_rotor, 10)

# =============================================================================
# TIPspl = csdl.power(10, BPM.TIP()/10)
# =============================================================================

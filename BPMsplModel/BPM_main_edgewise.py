import numpy as np
import csdl_alpha as csdl 
import pickle
from BPM_spl_model import BPMsplModel

recorder = csdl.Recorder(inline = True)
recorder.start()

def obs_time_computation():
    obs_t0 = 10*azimuth/(np.pi*rpm)   # initial guess for Newton method
    alpha = alpha*180/(2*np.pi)
    
    obs_t = csdl.ImplicitVariable(name='t', value=obs_t0)

    residual_1 = t - (((X - r*csdl.cos(azimuth)*csdl.cos(alpha)) + U*(obs_t - tau))**2 + (Y - r*csdl.sin(azimuth)*csdl.cos(alpha))**2 + (Z - r*csdl.cos(azimuth)*csdl.sin(alpha))**2)**(0.5) - tau
    residual_2 = 1 - (1/c0)*(((X - r*csdl.cos(azimuth)*csdl.cos(alpha)) + U*(obs_t - tau))**2 + (Y - r*csdl.sin(azimuth)*csdl.cos(alpha))**2 + (Z - r*csdl.cos(azimuth)*csdl.sin(alpha))**2)**(-0.5)*(X - r*csdl.cos(azimuth)*csdl.cos(alpha) + U*(obs_t - tau))*U
    
    solver = csdl.nonlinear_solver.Newton('solver_for_observerT')
    solver.add_state(obs_t, residual_1)
    solver.run()
    
    solver.add_state(obs_t, residual_2)
    solver.run()
    
    return obs_t

#load input for Edgewise condition from HJ                                     
with open('EdgewiseInput', 'rb') as f:
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
freq = Edgewise_input['f']

num_nodes = 1
num_radial = len(azimuth) # should be len(radial) (0.022 ~ 0.14[m] by 40 sections)
num_tangential = len(azimuth)
num_azim = num_tangential
num_observers = len(Edgewise_input['X'])
num_freq = len(freq)   
num_blades = Edgewise_input['B'].astype('int')
# num_blades = Edgewise_input['B']
                
mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

# Distance btw observer to blade section ref. from HJ
obs_x = Edgewise_input['X']
obs_y = Edgewise_input['Y']
obs_z = Edgewise_input['Z']

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
BPM = BPMsplModel(Edgewise_input, obs_position_tr, num_observers, num_radial, num_tangential, num_freq, num_nodes = 1)

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

U = Edgewise_input['V0']
c0 = Edgewise_input['a']
Mr = csdl.expand(U/c0, target_shape, 'ij->aijbc')
W = 1 + Mr*(x_tr/S_tr)

Spp_func = (2*np.pi/num_azim)*(W**2)*Spp_bar      # Spp_func = (W**2)*Spp_bar  # ok
Spp_R = num_blades*(1/(2*np.pi))*csdl.sum(Spp_func, axes=(3,))

Spp_rotor = csdl.sum(Spp_R, axes=(2,))

SPL_rotor = 10*csdl.log(Spp_rotor, 10)
OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

print('OASPL : ', OASPL.value)
    

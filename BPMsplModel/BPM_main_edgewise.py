import numpy as np
from lsdo_acoustics.core.acoustics import Acoustics
import csdl_alpha as csdl 
import pickle
from obs_tutorial_SC_2 import SteadyObserverLocation  # Import 'obs dist' using 'Class'
from BPM_spl_test2 import BPMsplModel

recorder = csdl.Recorder(inline = True)
recorder.start()
  
class DummyMesh(object):
    def __init__(self, num_radial, num_tangential):
        self.parameters = {
            'num_radial': num_radial,
            'num_tangential': num_tangential,
            'mesh_units': 'm'
        }

# load Input information
with open('input_data.pickle', 'rb') as f:
    input_data = pickle.load(f)

a = Acoustics(aircraft_position=np.array([0., 0., 0.,]))

a.add_observer(
    name='observer',
    obs_position=input_data['obs_loc'],
    time_vector=np.array([0.])
)

observer_data = a.assemble_observers()
velocity_data = np.array([0.,0.,0.])  # Q1 : steady -> velocity = 0 ?
observer_data['name']

obs = SteadyObserverLocation(observer_data, velocity_data)
steady_observer = obs.steady_observer_model()

num_nodes = 1 
num_observers = observer_data['num_observers']
num_radial = 5
num_cases = 5
num_tangential = 1
num_azim = num_tangential
num_blades = input_data['num_blades']

mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

# 1: Compute observer time (retarded time) using csdl.nonlinear solver



# Rotor coordinate transformation
propeller_radius = input_data['radius']
non_dim_r = csdl.Variable(value = np.linspace(0.2, 1., num_radial))
non_dim_rad_exp = csdl.expand(non_dim_r, (num_nodes, num_radial), 'i->ai')  #(num_nodes, num_radial)

radial_dist = csdl.expand(propeller_radius, (num_radial,)) * non_dim_r
twist_profile = csdl.Variable(value = 0.*np.pi/180, shape = (num_radial,1))

obs_position_tr = compute_rotor_frame_position(steady_observer, radial_dist, twist_profile, num_radial, num_azim, num_nodes, num_observers)

x_r = obs_position_tr['x_r']
y_r = obs_position_tr['y_r']
z_r = obs_position_tr['z_r']
S = (x_r)**2 + (y_r)**2 + (z_r)**2
            
# Computation total BPM spl with respect to 4 sub-models
BPM = BPMsplModel(observer_data, input_data, obs_position_tr, num_radial, num_tangential)

TBLTEspl = csdl.power(10, BPM.TBLTE()/10)
LBLVSspl = csdl.power(10, BPM.LBLVS()/10)
TIPspl = csdl.power(10, BPM.TIP()/10)
BLUNTspl = csdl.power(10, BPM.BLUNT()/10)
BPMsum = TBLTEspl + LBLVSspl + TIPspl + BLUNTspl  

totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20
Spp_bar = csdl.power(10, totalSPL/10) # previous note: shape is (num_nodes, num_observers, num_radial, num_azim)

# For "Hover" condition - Ref. HJ
rpm = input_data['RPM'][0]
         
target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades)
U = non_dim_rad_exp * (2*np.pi/60.) * rpm * csdl.expand(propeller_radius, (num_nodes, num_radial)) #(num_nodes, num_radial)
u = csdl.expand(U, target_shape, 'ij->iajbc')
c0 = csdl.expand(0.34, target_shape)   # arbitrary value : c0 = sound_of_speed
Mr = u/c0 

# Expansion of Distance from obs to rotor
x_r = csdl.expand(x_r, target_shape, 'ijkl->ijkla')
S = csdl.expand(S, target_shape, 'ijkl->ijkla')

# Spp_bar expansion
Spp_bar_exp = csdl.expand(Spp_bar, target_shape, 'ijkl->ijkla')

W = 1 + Mr*x_r/S # shape is (num_nodes, num_observers, num_radial, num_azim, num_blades)
Spp = csdl.sum(Spp_bar_exp*(W**2), axes=(3,)) * (num_azim/(2*np.pi)) #Q: Does axes(3,) mean summation with respect to num_azim 
Spp_R = csdl.sum(Spp, axes=(2,))

rotorSPL = 10*csdl.log(csdl.sum(Spp_R, axes=(2,)))      #Q: Spp summation with option axes = (2,), with respect to number of blades(n)?


# 2: Compute new global location of observer to rotor hub
# 3: Coordinate transformation observer to local blade section
# 4: Compute SPl total


def compute_rotor_frame_position(steady_observer, radial_dist, twist_profile, num_radial, num_azim, num_nodes, num_observers):
    rel_obs_x_pos = steady_observer['rel_obs_x_pos']
    rel_obs_y_pos = steady_observer['rel_obs_y_pos']
    rel_obs_z_pos = steady_observer['rel_obs_z_pos']

    target_shape = (num_nodes, num_observers, num_radial, num_azim)
    
    azim_angle = np.linspace(0, 2*np.pi, num_azim+1)[:-1]
    azim_dist = csdl.Variable(value = azim_angle)
    
    x_exp = csdl.expand(rel_obs_x_pos, target_shape)
    y_exp = csdl.expand(rel_obs_y_pos, target_shape)
    z_exp = csdl.expand(rel_obs_z_pos, target_shape)
    
    twist_exp = csdl.expand(twist_profile, target_shape, 'ij->abij')
    radius_exp = csdl.expand(radial_dist, target_shape, 'i->abic')
    azim_exp = csdl.expand(azim_dist, target_shape, 'i->abci')
    
    sin_th = csdl.sin(twist_exp)
    cos_th = csdl.cos(twist_exp)
    sin_ph = csdl.sin(azim_exp)
    cos_ph = csdl.cos(azim_exp)
    
    beta_p = 0.   # flapping angle
    coll = 0.     # collective pitch
    
    x_r = x_exp*cos_th*sin_ph - y_exp*cos_ph*cos_th + (z_exp+radius_exp)*sin_th
    y_r = x_exp*cos_ph + y_exp*sin_th
    z_r = -x_exp*sin_ph*sin_th + y_exp*cos_ph*sin_th + (z_exp+radius_exp)*cos_th
    
    obs_position_tr = {
        'x_r': x_r,
        'y_r': y_r,
        'z_r': z_r
        }
    
    return obs_position_tr
                                 
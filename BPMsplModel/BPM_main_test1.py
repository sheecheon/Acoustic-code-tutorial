import numpy as np
from lsdo_acoustics.core.acoustics import Acoustics
import csdl_alpha as csdl 
import pickle
from obs_tutorial_SC_2 import SteadyObserverLocationTutorial  # Import 'obs dist' using 'Class'

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

obs = SteadyObserverLocationTutorial(observer_data, velocity_data)
SteadyObserver = obs.steady_observer_model()


num_radial = 5 

mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

num_cases = 5
num_tangential = 1

# for i in range(num_cases):
#     # sim_skm['rpm'] = input_data['RPM'][i]
#     # sim_skm['chord_profile'] = chord*np.ones((num_radial,))
#     # sim_skm['propeller_radius'] = input_data['radius']
#     # sim_skm['CT'] = input_data['CT'][i]
#     # sim_skm.run()
#     # skm_noise.append(sim_skm['broadband_spl'][0][0])

#     sim_gl['rpm'] = input_data['RPM'][i]
#     sim_gl['chord_profile'] = chord*np.ones((num_radial,))
#     sim_gl['propeller_radius'] = input_data['radius']
#     sim_gl['CT'] = input_data['CT'][i]
#     sim_gl.run()
#     gl_noise.append(sim_gl['broadband_spl'][0][0])

#     # # SKM ERRORS
#     # skm_HJ_error.append((HJ_SKM[i] - skm_noise[i]) / HJ_SKM[i])
#     # skm_exp_error.append((exp_data[i] - skm_noise[i]) / exp_data[i])

#     # GL ERRORS
#     gl_HJ_error.append((HJ_GL[i] - gl_noise[i]) / HJ_GL[i])
#     gl_exp_error.append((exp_data[i] - gl_noise[i]) / exp_data[i])

#     print(gl_exp_error)
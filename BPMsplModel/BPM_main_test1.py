import numpy as np
from lsdo_acoustics.core.acoustics import Acoustics
import csdl_alpha as csdl 
import pickle
from obs_tutorial_SC_2 import SteadyObserverLocation  # Import 'obs dist' using 'Class'
from BPM_spl_test2 import BPMsplModel

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
SteadyObserver = obs.steady_observer_model()

num_radial = 5 

mesh = DummyMesh(
    num_radial=num_radial,
    num_tangential=1 # this input is useless but kept for now in case it's needed in the future
)

num_cases = 5
num_tangential = 1

BPM = BPMsplModel(observer_data, input_data, SteadyObserver, num_radial, num_tangential)

TBLTEspl = BPM.TBLTE()
LBLVSspl = BPM.LBLVS()
TIPspl = BPM.TIP()
BLUNTspl = BPM.BLUNT()


# class BPMsplModel():
#     def __init__(self, observer_data, input_data, SteadyObserver, num_radial, num_tangential, num_nodes = 1):

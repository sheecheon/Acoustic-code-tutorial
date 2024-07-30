import numpy as np
import csdl_alpha as csdl 
import pickle
from BPM_spl_model import BPMsplModel

recorder = csdl.Recorder(inline = True)
recorder.start()

#load input for Edgewise condition from HJ                                     
with open('Edgewise_input', 'rb') as f:
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
freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000,25000,31500,40000,50000]) # 1/3 octave band central frequency [Hz]


c0 = Edgewise_input['a']

azimuth = Edgewise_input['azimuth']
rpm = Edgewise_input['RPM']
alpha = Edgewise_input['alpha']
Psi = Edgewise_input['Psi']

r = Edgewise_input['R']       #sectional radius
M = Edgewise_input['M']
U = M*c0

X = Edgewise_input['X']
Y = Edgewise_input['Y']
Z = Edgewise_input['Z']

# Variable expansion
num_nodes = 1
num_observers = 1
num_radial = 40         #len(radial)
num_azim = len(azimuth)
num_blades = 2    # dtype should be modified!!!
num_freq = len(freq)
target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq, num_blades)

alpha = csdl.expand(alpha, target_shape)
azimuth = csdl.expand(azimuth, target_shape, 'ij->abcidj')
Psi = csdl.expand(Psi, target_shape, 'ij->aijbcd')

X_exp = csdl.expand(X, target_shape)
Y_exp = csdl.expand(Y, target_shape)
Z_exp = csdl.expand(Z, target_shape)

r = csdl.expand(r, target_shape, 'ij->iajbcd')

# Preprocess for Input
alpha = alpha*180/(2*np.pi)
tau = 30*azimuth/(np.pi*rpm)
obs_t0 = 30*Psi/(np.pi*4047)   # obs_t0 = 30*Psi/(np.pi*rpm)   # initial guess for Newton method
obs_t = csdl.ImplicitVariable(name='obs_t', value = obs_t0.value)   #0.04483238

# # ====================== check for the 'out of memory' ========================
# X_exp = 0.
# Y_exp = 2.2755
# Z_exp = -2.7188
# X_exp = X_exp[0, 0, -1, -1, -1, 0]
# Y_exp = Y_exp[0, 0, -1, -1, -1, 0]
# Z_exp = Z_exp[0, 0, -1, -1, -1, 0]
# azimuth = azimuth[0, 0, -1, -1, -1, 0]
# alpha = alpha[0, 0, -1, -1, -1, 0]
# obs_t = obs_t0[0, 0, -1, -1, -1, 0]
# r = r[0, 0, -1, -1, -1, 0]
# tau = tau[0, 0, -1, -1, -1, 0]
# obs_t = csdl.ImplicitVariable(name='obs_t', value=obs_t.value)     #0.04483238 
# # =============================================================================

X_exp = X_exp[0, 0, -1, -1, :5, 0]
Y_exp = Y_exp[0, 0, -1, -1, :5, 0]
Z_exp = Z_exp[0, 0, -1, -1, :5, 0]
azimuth = azimuth[0, 0, -1, -1, :5, 0]
alpha = alpha[0, 0, -1, -1, :5, 0]
obs_t = obs_t0[0, 0, -1, -1, :5, 0]
r = r[0, 0, -1, -1, :5, 0]
tau = tau[0, 0, -1, -1, :5, 0]
obs_t = csdl.ImplicitVariable(name='obs_t', value=obs_t.value)     #0.04483238 

# obs_t = csdl.expand(0.04483238, target_shape)
residual_1 = obs_t - ((((X_exp - r*csdl.cos(azimuth)*csdl.cos(alpha)) + U*(obs_t - tau))**2 + (Y_exp - r*csdl.sin(azimuth)*csdl.cos(alpha))**2 + (Z_exp - r*csdl.cos(azimuth)*csdl.sin(alpha))**2)**(0.5))/c0 - tau
residual_2 = 1 - (1/c0)*(((X_exp - r*csdl.cos(azimuth)*csdl.cos(alpha)) + U*(obs_t - tau))**2 + (Y_exp - r*csdl.sin(azimuth)*csdl.cos(alpha))**2 + (Z_exp - r*csdl.cos(azimuth)*csdl.sin(alpha))**2)**(-0.5)*(X_exp - r*csdl.cos(azimuth)*csdl.cos(alpha) + U*(obs_t - tau))*U

solver = csdl.nonlinear_solvers.Newton('solver_for_observerT')
solver.add_state(obs_t, residual_1)
solver.run()
print(obs_t.value)
# print(obs_t0)
print(residual_1.value)


# solver.add_state(obs_t, residual_2)
# solver.run()

# return obs_t
  

# csdl.nonlinear_solvers.Newton()
# csdl.nonlinear_solvers.GaussSeidel()
# csdl.nonlinear_solvers.Jacobi()
# csdl.nonlinear_solvers.BracketedSearch()
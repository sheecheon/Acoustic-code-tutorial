import numpy as np
import csdl_alpha as csdl
import pickle
from lsdo_acoustics.core.acoustics import Acoustics
from obs_tutorial_SC import steady_observer_model  # using fuction, not class

# load Input information
with open('input_data.pickle', 'rb') as f:
    input_data = pickle.load(f)

a = Acoustics(aircraft_position = np.array([0., 0., 0.]))

obs_radius = 1.5
num_observers = 37
theta = np.linspace(0, np.pi, num_observers)
z = obs_radius * np.cos(theta)
x = obs_radius * np.sin(theta)
obs_position_array = np.zeros((num_observers, 3))
obs_position_array[:,0] = x
obs_position_array[:,2] = z

a.add_observer(
    name = 'observer',
    obs_position = np.array([1.859475, 0, -1.302018]),
    time_vector = np.array([0.]),
    obs_velocity = np.array([0., 0., 0.])
    )

observer_data = a.assemble_observers()
velocity_data = np.array([0.,0.,0.])  # Q1 : steady -> velocity = 0 ?
observer_data['name']

obs_velocity = np.array([0., 0., 0.])


nondim_sectional_radius = np.linspace(0.21, 0.99, 40)


#Q: Is it correct to obtain observer info. from my demo code?
rel_obs_dist, rel_angle_plane, rel_angle_normal =  steady_observer_model(observer_data, obs_velocity) 
# chord_profile = 
             
def lowson_spl_model(input_data, observer_data, num_nodes = 1):
    P_ref = 2.e-5

    # num_nodes = self.parameters['num_nodes']
    B = input_data['num_blades']
    num_observers = observer_data['num_observers']
    modes = [1., 2., 3.]
    num_modes = len(modes)
    harmonics = np.arrange(0, 11, 1)  # load_harmonics
    num_harmonics = len(harmonics)
    num_radial = len(chord_profile)
    
    a = 343 #speed of sound
    num_blades = 3
    RPM = 1500
    radius = 0.3556
    M = 0
    rho = 1.225
    A = np.pi*radius**2    #Dimension of the disk
    omega = RPM*2.*np.pi/60.
        
    thrust_dir = np.array([0., 0., 1.,])
    in_plane_ex = np.array([1., 0., 0.])
    r = nondim_sectional_radius    # Q : distance from observer to source?
    # r1 = #Q : distance from observer to hub? / r1 = self.convection_adjustment(S, x, y, z, Vx, Vy, Vz, a)

# =============== Q : Coord. transformation according to Fig. 3? ==================
# Projection the observer X-location along the x-direction of the aero coordinate system vector : Ref.

    x_dir_aero = csdl.Variable(value = in_plane_ex) #Q: old: self.declare_variable -> csdl.Variable, is it ok?
    thrust_dir = csdl.expand(thrust_dir, (num_nodes, 3, num_observers), 'i->aib')
    rel_obs_position = csdl.Variable(value = obs_position_array)
    
    z_in_frame = csdl.product(rel_obs_position, thrust_dir, axes(1,))  #old: csdl.dot
    
    
   # def convection_adjustment(self, S, x, y, z, Vx, Vy, Vz, a):
   #     '''
   #     Need to calculate the component of the Mach number in the direction of the observer
   #     '''
   #     num_nodes, num_observers = x.shape[0], x.shape[1]
   #     position = self.declare_variable('rel_obs_position', shape=(num_nodes, 3, num_observers))
   #     # position = self.create_output('obs_pos', shape=(num_nodes, num_observers, 3))
   #     velocity = self.create_output('aircraft_vel', shape=(num_nodes, 3, num_observers))

   #     # position[:,:,0] = csdl.reshape(x, (num_nodes, num_observers, 1))
   #     # position[:,:,1] = csdl.reshape(y, (num_nodes, num_observers, 1))
   #     # position[:,:,2] = csdl.reshape(z, (num_nodes, num_observers, 1))

   #     velocity[:,0,:] = csdl.reshape(Vx, (num_nodes, 1, num_observers))
   #     velocity[:,1,:] = csdl.reshape(Vy, (num_nodes, 1, num_observers))
   #     velocity[:,2,:] = csdl.reshape(Vz, (num_nodes, 1, num_observers))

   #     v_comp_obs = csdl.dot(velocity, position, axis=1) / S

   #     r1 = S*(1-v_comp_obs/csdl.expand(a, v_comp_obs.shape))
   #     return r1 # shape is (num_nodes, num_observers)





# =========================== Variable expansion ==============================


    R = csdl.expand(propeller_radius, ,,)           
    
# =============================================================================
# This sign will be replaced by 'csdl loops', arbitrary value for temporary
    n = np.ones(shape=coeff_target_shape)
    lam = np.ones(shape = coeff_target_shape)
    n_var = self.create_input('n_var', val=n)
    lam_var = self.create_input('lam_var', val=lam)

    term_1_coeff_A = 1
    term_2_coeff_A = -1
    term_1_coeff_B = 1
    term_1_coeff_B = -1
    
    coeff_sign_matrix_odd = 1
    coeff_sign_matrix_even = 1
# =============================================================================
    # term A
    term_1_constant = n_var*omega_exp*z_exp/(a_exp*r1_exp**2)
    term_2_constant = 1./(R_exp*r_ex*r1_exp) #Q : why r is used with r1?
    bessel_input = n_var*omega_exp*R_exp*r_exp*Y_exp/(a_exp*r1_exp) # Q :r
    
    term_1_A_fc = (coeff_sign_matrix_even * b_T_uns_exp + coeff_sign_matrix_odd * a_T_uns_exp)
    term_2_A_fc = (coeff_sign_matrix_even * b_D_uns_exp + coeff_sign_matrix_odd * a_D_uns_exp)
    term_1_A = term_1_constant * term_1_A_fc * (csdl.bessel(bessel_input, order = n-lam) + \
    A_lin_comb_sign_matrix * csdl.power(-1., lam_var) * csdl.bessel(bessel_input, order = n-lam))
    term_2_A = term_2_constant * term_2_A_fc * ((n_var - lam_var) * csdl.bessel(bessel_imput, order = n-lam) + \
    A_lin_comb_sign_matrix * csdl.power(-1., lam_var) * (n_var + lam_var) * csdl.bessel(bessel_imput, order = n+lam))
        
    a_n_radial_harmonics = (term_1_coeff_A * term_1_A + term_2_coeff_A * term_2_A)/(4*np.pi)
    
    # term B
    term_1_B_fc = (coeff_sign_matrix_even * a_T_uns_exp + coeff_sign_matrix_odd * b_T_uns_exp) # weighting based on sign of n-lambda
    term_2_B_fc = (coeff_sign_matrix_even * a_D_uns_exp + coeff_sign_matrix_odd * b_D_uns_exp) # weighting based on sign of n-lambda
    term_1_B = term_1_constant * term_1_B_fc * (csdl.bessel(bessel_input, order = n-lam) + \
    B_lin_comb_sign_matrix * csdl.power(-1., lam_var) *csdl.bessel(bessel_input, order = n+lam))
    term_2_B = term_2_constant * term_2_B_fc * ((n_var-lam_var) * csdl.bessel(bessel_input, order = n-lam) + \
    B_lin_comb_sign_matrix * csdl.power(-1., lam_var) * (n_var+lam_var) * csdl.bessel(bessel_input, order = n+lam))

    b_n_radial_harmonics = (term_1_coeff_B*term_1_B + term_2_coeff_B*term_2_B)/(4*np.pi)
    
    # Note for 'axes' option : target_shape = (num_nodes, num_observers, num_nodes, B, num_harmonics, num_radial)
    a_n_radial = csdl.sum(a_n_radial_harmonics, axes = (4,))  # summation over harmonics (lam)
    b_n_radial = csdl.sum(b_n_radial_harmonics, axes = (4,)) 

    An = csdl.sum(a_n_radial, axes = (4,))  #Q : According to the equ.4 from ref. HJ, summation should be along blade span (N), axes =4?
    Bn = csdl.sum(b_n_radial, axes = (4,))
    sum_AB = (An)**2 + (Bn)**2
    
    bladeSPL = 10.*csdl.log(sum_AB/(2*(P_ref**2)), 10)
    
    ex = csdl.power(10., bladeSPL/10.)
    
    rotorSPL = 10.*csdl.log(csdl.sum(ex, axes = (3,)), 10.)  #SPL_m, and equ.6 (last equation) in ref.
    
    #rotor_tonal_spl_uns = 10*csdl.log10(csdl.sum(csdl.exp_a(10.,SPL_m/10.), axes=(2,)))  #Q:
 
  
    
  
    
  
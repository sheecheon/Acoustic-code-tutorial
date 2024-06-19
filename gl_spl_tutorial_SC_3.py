import numpy as np
import csdl_alpha as csdl
import pickle
from lsdo_acoustics.core.acoustics import Acoustics
from obs_tutorial_SC import steady_observer_model  # using fuction, not class

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

rel_obs_dist, rel_angle_plane, rel_angle_normal =  steady_observer_model(observer_data, velocity_data)
# rel_obs_dist = 2.7

freq_band = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
          500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
          10000, 12500, 16000, 20000,
          25000, 31500, 40000, 50000, 63000])# additional ones used by Hyunjune  # default value

# recorder = csdl.Recorder(inline=True)
# recorder.start()

def gl_spl(observer_data, input_data, freq_band, rel_obs_dist, num_nodes=1):
    
    # Information from 'test_broadband_validation'
    B = input_data['num_blades']
    R = input_data['radius']
    CT = input_data['CT'][0]
    rpm = input_data['RPM'][0]

    # Mach = np.zeros_like(CT)
    chord = input_data['chord']
    
    num_observers = observer_data['num_observers']
    frequency_band = freq_band
    num_freq_band = len(freq_band)
    
    num_radial=5
    dr = (1 - 0.2) * R / (num_radial-1)
    a = 343 # speed of sound 
    chord_profile = chord*np.ones((num_radial,))
    theta_0 = 0.*np.pi/180
    
    S = rel_obs_dist.value       #Q: rel_obs_dist.value instead of rel_obs_dist is correct? -> 
    omega = rpm*2*np.pi/60.
    # V_aircraft = csdl.Variable(value = np.array([0., 0., 0.,]))
    V_aircraft = np.array([0., 0., 0])
    # V_inf = csdl.norm(V_aircraft, axes=(1,))
    V_inf = csdl.norm(V_aircraft)  # norm axis define?,,
    V_tip = omega*R - V_inf
    A_b = csdl.sum(chord_profile)*dr
    sigma = A_b*B/(np.pi*R**2)
    c_sw = sigma*np.pi*R/B
            
    V_t = V_tip
    M_t = V_t/a
    St = frequency_band*c_sw/V_t
            
    # f0 = csdl.log10(V_t**7.84)*10.
    # f1 = sigma
    # f2 = 0.9*M_t*sigma*(M_t+3.82)
    # f3 = 1. # NOT USED
    # f4 = 1. # NOT USED
    # f5 = -2.*M_t**2+2.06
    # f6 = -CT*M_t*(CT-csdl.sin((theta_0**2)**0.5)+2.06)+1.
    # f7 = CT
    # f8 = 4.97*CT*csdl.sin((theta_0**2)**0.5)*(1.5*S/R*M_t - (S/R) + 15.)
    f0 = csdl.log(V_t**7.94,10)*10.
    f1 = sigma
    f2 = 0.9*M_t*sigma*(M_t+3.82)
    f3 = 1. # NOT USED
    f4 = 1. # NOT USED
    f5 = -2.*M_t**2+2.06
    f6 = -CT*M_t*(CT-csdl.sin((theta_0**2)**0.5)+2.06)+1.
    f7 = CT
    f8 = 4.97*CT*csdl.sin((theta_0**2)**0.5)*(1.5*S/R*M_t - (S/R) + 15.)
    
    # num = f0*(St-(f1*csdl.log10(CT) + f2*csdl.log10(sigma)))**0.6
    # den_1 = csdl.exp_a(
    #      f3*(St-(f1*csdl.log10(CT)+f2*csdl.log10(sigma)) + f5),
    #      f6
    # )
    # den_2 = csdl.exp_a(
    #     f7*(St-(f1*csdl.log10(CT)+f2*csdl.log10(sigma))),
    #     f8
    # )
    num = f0*(St-(f1*csdl.log(CT,10) + f2*csdl.log(sigma,10)))**0.6
    den_1 = (f3*(St-(f1*csdl.log(CT,10)+f2*csdl.log(sigma,10)) + f5))**f6
    
    den_2 = (f7*(St-(f1*csdl.log(CT,10)+f2*csdl.log(sigma,10))))**f8
            
    SPL_1_3 = num/(den_1 + den_2)
    
    OASPL = 10.*csdl.log(
          csdl.sum(
              csdl.power(10., SPL_1_3/10.),
              # axes=(0,) # OVER THE FREQUENCY AXIS
          ),10
      )
            
    return OASPL
        
# def obs_distance

recorder = csdl.Recorder(inline=True)
recorder.start()
GLspl = gl_spl(observer_data, input_data, freq_band, rel_obs_dist)
print('GL spl is:', GLspl.value)
recorder.stop()

 
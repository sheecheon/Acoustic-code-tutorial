import numpy as np
import csdl_alpha as csdl
from lsdo_acoustics.core.acoustics import Acoustics
from revised.BPM_model import BPMVariableGroup, BPM_model

# =============================================================================
# class DummyMesh(object):
#     def __init__(self, num_radial, num_tangential):
#         self.parameters = {
#             'num_radial': num_radial,
#             'num_tangential': num_tangential,
#             'mesh_units': 'm'
#         }
# 
# # observer data
# a = Acoustics(aircraft_position=np.array([0., 0., 0.,]))
# 
# obs_radius = 1.5
# num_observers = 37
# theta = np.linspace(0, np.pi, num_observers)
# z = obs_radius * np.cos(theta)
# x = obs_radius * np.sin(theta)
# 
# obs_position_array = np.zeros((num_observers, 3))
# obs_position_array[:,0] = x
# obs_position_array[:,2] = z
# 
# for i in range(num_observers):
#     a.add_observer(
#         name=f'obs_{i}',
#         obs_position=obs_position_array[i,:],
#         time_vector=np.array([0.])
#     )
# observer_data = a.assemble_observers()


# ==================== Blade condition : 'Ideal twist' ========================
flight_condition = 'hover'

x = 1.8595
y = 0
z = -1.3020
num_observers = 1
observer_data = {'x': x,
                 'y': y,
                 'z': z,
                 'num_observers': num_observers
                 }

num_blades = 4
RPM = 5465
radius = 0.3556

M = 0
rho = 1.225 # air density
mu = 1.789*(1e-5) # dynamic_viscosity
c0 = 340.3  # sound of speed
omega = RPM*2.*np.pi/60.
Vinf = M*c0

freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                 6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]
num_freq = len(freq)

chord = 0.03176
radial = np.linspace(0.033, 0.157, 40)
sectional_span = radial[2] - radial[1]
num_radial = len(radial)
azimuth = np.linspace(0, 2*np.pi, 40)
num_azim = len(azimuth)
num_nodes = 1 ## 

AOA = np.array([3.1870,	4.8630,	5.5070, 5.7190, 5.7150,	
                5.6030, 5.4370, 5.2490, 5.0550, 4.8640,
                4.6790, 4.5030, 4.3370, 4.1810, 4.0340,
                3.8960, 3.7670, 3.6460, 3.5320, 3.4240, 
                3.3230, 3.2270, 3.1360, 3.0510, 2.9690, 
                2.8920, 2.8180, 2.7480, 2.6800, 2.6150,	 
                2.5500, 2.4870, 2.4220, 2.3540,	 2.2780,	
                2.1900, 2.0760, 1.9180, 1.6670, 1.1480])
                
A_cor = 0.
a_star = AOA + A_cor
                
V0 = np.array([20.5490, 22.2650, 24.0070, 25.7630, 27.5310,
               29.3070, 31.0900, 32.8770, 34.6690, 36.4640,
               38.2610, 40.0610, 41.8630, 43.6660, 45.4720,
               47.2790, 49.0870, 50.8970, 52.7070, 54.5190,
               56.3320, 58.1450, 59.9600, 61.7750, 63.5910,
               65.4070, 67.2250, 69.0430, 70.8610, 72.6800,
               74.5000, 76.3200, 78.1410, 79.9640, 81.7880,
               83.6140, 85.4450, 87.2820, 89.1330, 91.0260])


pitch = np.array([33.3352534781022, 29.6213069376725, 27.4020828999388, 25.5121152576512,
                  23.7770455253560, 22.3411872524742, 21.0055166711821, 19.5071423853625, 
                  18.4805359215146, 17.5647444659885, 16.7947472538093, 16.0499336349242, 
                  15.3317507026988, 14.5144734768082, 13.9724023102524, 13.4522735371309, 
                  12.9572239545054, 12.4951500965938, 11.9345829307194, 11.5524634047052,
                  11.1948614516058, 10.8569121935810, 10.5401256647001, 10.2465955360402, 
                  9.88588384521158, 9.62932663145703, 9.37940365343519, 9.13244725266795, 
                  8.88668384685799, 8.64953433908840, 8.36393233157182, 8.17679116478389, 
                  8.00651695665029, 7.84678637378259, 7.69506422422990, 7.50702752273457, 
                  7.37366526471621, 7.24068761858740, 7.10235778665474, 6.95293897122480])

alpha = 0.
sigma = 0.255
TE_thick = 8.e-04   # h
slope_angle = 16.  #Psi

recorder = csdl.Recorder(inline = True)
recorder.start()

BPM_vg = BPMVariableGroup(
    chord=chord,
    radial=radial, #R
    sectional_span=sectional_span, # R(1) - R(2) / l
    a_star=a_star, # AoAcor = 0. / a_star = AOA - AOAcor.
    pitch=pitch,
    azimuth=azimuth,
    alpha=alpha,
    RPM = RPM,
    TE_thick=TE_thick, #h
    slope_angle=slope_angle, #Psi
    free_vel=V0, #free-stream velocity U and V0
    freq=freq,
    RPM=RPM,
    speed_of_sound=c0, #c0
    Vinf = Vinf,
    density=rho,  #rho
    dynamic_viscosity=mu,  #mu
    num_radial=num_radial,
    num_tangential=num_azim,   # num_tangential = num_azim
    num_freq=num_freq
    )
                          
TBLTE_dep, spl_BLUNT, spl_LBLVS, observer_data_tr = BPM_model(BPMVariableGroup = BPM_vg,
                                                              observer_data = observer_data,
                                                              num_observers = observer_data['num_observers'],
                                                              num_blades = num_blades,
                                                              num_nodes = 1,
                                                              flight_condition = flight_condition
                                                              )
splp = TBLTE_dep['splp']
spls = TBLTE_dep['spls']
spla = TBLTE_dep['spla']

BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) # + csdl.power(10, spl_LBLVS) : untripped condition
totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20

# ======================= Final Rotor spl computation =========================
x_tr = observer_data_tr['x_tr']
S_tr = observer_data_tr['obs_dist_tr']

target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq)
Mr = V0/c0
exp_Mr = csdl.expand(Mr, target_shape, 'i->abicd')  

W = 1 + exp_Mr*(x_tr/S_tr)

# =================== Intial computation for rotor SPL ========================
Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)
Spp_func = (2*np.pi/(num_azim-1))*(W**2)*Spp_bar      # Spp_func = (W**2)*Spp_bar  # ok
Spp_R = num_blades*(1/(2*np.pi))*csdl.sum(Spp_func, axes=(3,))
Spp_rotor = csdl.sum(Spp_R, axes=(2,))

SPL_rotor = 10*csdl.log(Spp_rotor, 10)
OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

print('OASPL : ', OASPL.value)
# A = SPL_rotor[0, 0, :].value

# =================== HJ's ref. data for Verification =========================
BPM_HJ_SPL_rotor = np.array([4.01416375678111,	12.3925109923754,	19.4054345603941,	26.1178174659802,
                             32.1179818337257,	37.3382567246601,	42.0638086876374,	45.4988580903863,
                             48.4897383843208,	51.2108063408457,	53.3698213559171,	54.3574453088175,    
                             54.3633907171668,	54.1522962158025,	55.9967811066713,	60.3966480873478,
                             52.8219456450066,	45.3882941378738,	41.8417569668817,	38.0095878598943,
                             33.0663604621407,	27.1784549607477])

BPM_HJ_OASPL = 64.8128

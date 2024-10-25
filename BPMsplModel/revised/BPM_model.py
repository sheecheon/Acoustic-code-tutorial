# 09. 17. 2024: BPM spl model refinement 
import numpy as np
import csdl_alpha as csdl
from dataclasses import dataclass
from csdl_alpha.utils.typing import VariableLike, Variable
from revised.disp_thick import disp_thick
from revised.BPM_spl_model import TBLTE, TE_BLUNT, LBLVS
from revised.rotor_spl_computation import rotor_spl_computation
from scipy import io


@dataclass
class BPMVariableGroup(csdl.VariableGroup):
    chord: VariableLike
    radial: VariableLike #R
    sectional_span: VariableLike # R(1) - R(2) / l
    a_star: VariableLike # AoAcor = 0. / a_star = AOA - AOAcor.
    
    pitch: VariableLike
    azimuth: VariableLike
    alpha: VariableLike
    TE_thick: VariableLike  #h
    slope_angle: VariableLike #Psi
    
    free_vel: VariableLike #free-stream velocity U and V0
    freq: VariableLike

    RPM: VariableLike
    speed_of_sound: VariableLike #c0
    Vinf: VariableLike
    density: VariableLike  #rho
    dynamic_viscosity: VariableLike  #mu

    num_radial: int
    num_tangential: int   # num_tangential = num_azim
    num_freq: int
 
def BPM_model(BPMVariableGroup, observer_data, num_blades, num_nodes, flight_condition, outband):
    
    num_observers = observer_data['num_observers']
    num_radial = BPMVariableGroup.num_radial
    num_azim = BPMVariableGroup.num_tangential
    num_freq = BPMVariableGroup.num_freq
    
    chord = BPMVariableGroup.chord
    R = BPMVariableGroup.radial
    L = BPMVariableGroup.sectional_span
    a_star = BPMVariableGroup.a_star
    
    radial = BPMVariableGroup.radial
    azimuth = BPMVariableGroup.azimuth
    pitch = BPMVariableGroup.pitch
    alpha = BPMVariableGroup.alpha
    RPM = BPMVariableGroup.RPM
    TE_thick = BPMVariableGroup.TE_thick
    slope_angle = BPMVariableGroup.slope_angle
    
    c0 = BPMVariableGroup.speed_of_sound
    rho = BPMVariableGroup.density
    mu = BPMVariableGroup.dynamic_viscosity
    
    U = BPMVariableGroup.free_vel
    c0 = BPMVariableGroup.speed_of_sound
    
    f = BPMVariableGroup.freq
    
    
    if flight_condition == 'hover':            
        #========================= Variable expansion =========================
        target_shape = (num_nodes, num_observers, num_radial, num_azim, num_freq)
        a_star = csdl.expand(a_star, target_shape, 'i->abicd')
        u = csdl.expand(U, target_shape, 'i->abicd')
        chord = csdl.expand(chord, target_shape)
        f = csdl.expand(f, target_shape, 'i->abcdi')
        
        sectional_mach = u/c0
        machC = 0.8*sectional_mach
        Rc = rho*u*chord/mu   #sectional Reynolds number
        l = csdl.expand(L, target_shape)
        
        #====================== Displacement thickness ========================
        DT_s, DT_p, BT_p = disp_thick(a_star, Rc, chord)
        
        #=========== observer coordinate transformation for 'Hover' ===========
        obs_x = observer_data['x']   
        obs_y = observer_data['y']
        obs_z = observer_data['z']
        
        # with multiple observer
        exp_x = csdl.expand(obs_x, target_shape,'i->aibcd')    
        exp_y = csdl.expand(obs_y, target_shape,'i->aibcd')   
        exp_z = csdl.expand(obs_z, target_shape,'i->aibcd')   
    
        exp_azim = csdl.expand(azimuth, target_shape, 'i->abcid')
        exp_radial = csdl.expand(csdl.reshape(radial, (num_radial,)), target_shape, 'i->abicd')
    
        xloc = exp_radial*csdl.cos(exp_azim)
        yloc = exp_radial*csdl.sin(exp_azim)
    
        sectional_x = exp_x - xloc    #newX
        sectional_y = exp_y - yloc    #newY
        sectional_z = exp_z           
        
        twist_exp = csdl.expand(csdl.reshape(pitch*(np.pi/180),(num_radial,)), target_shape, 'i->abicd')
        azim_exp = csdl.expand(azimuth, target_shape, 'i->abcid')
        
        beta_p = 0.   # flapping angle
        coll = 0.     # collective pitch
        
        sin_th = csdl.sin(twist_exp)
        cos_th = csdl.cos(twist_exp)
        sin_ph = csdl.sin(azim_exp)
        cos_ph = csdl.cos(azim_exp)
        # According to ref. HJ, M_beta = 0, last term does not account for this file
         
        x_tr = sectional_x*cos_th*sin_ph - sectional_y*cos_ph*cos_th + sectional_z*sin_th
        y_tr = sectional_x*cos_ph + sectional_y*sin_ph
        z_tr = -sectional_x*sin_th*sin_ph + sectional_y*cos_ph*sin_th + sectional_z*cos_th
        
        obs_dist_tr = ((x_tr)**2 + (y_tr)**2 + (z_tr)**2)**0.5
        
        observer_data_tr = {'num_observers': num_observers,
                            'x_tr': x_tr,
                            'y_tr': y_tr,
                            'z_tr': z_tr,
                            'obs_dist_tr': obs_dist_tr
                            }
        

    elif flight_condition == 'edgewise':
        #========================= Variable expansion =========================
        target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_freq)
        a_star = csdl.expand(a_star, target_shape, 'ijk->abjikc')
        u = csdl.expand(U, target_shape, 'ijk->abjikc')
        chord = csdl.expand(chord, target_shape,'i->abicde')
        f = csdl.expand(f, target_shape, 'i->abcdei')
        
        sectional_mach = u/c0
        machC = 0.8*sectional_mach
        Rc = rho*u*chord/mu   #sectional Reynolds number
        l = csdl.expand(L, target_shape)
        # ====================== Displacement thickness =======================
        DT_s, DT_p, BT_p = disp_thick(a_star, Rc, chord)
        
        #========= observer coordinate transformation for 'Edgewise' ==========
        # time_shape does not account for 'num_freq' due to memory comsuming issue for now
        time_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades)   
        obs_x = observer_data['x']
        obs_y = observer_data['y']
        obs_z = observer_data['z']
        
        RPM = BPMVariableGroup.RPM
        tau = 30*azimuth/(np.pi*RPM)   # source time
        alpha = alpha*(2*np.pi)/180

        Vinf = BPMVariableGroup.Vinf

        # with multiple observer
        exp_x = csdl.expand(obs_x, time_shape,'i->aibcd')    
        exp_y = csdl.expand(obs_y, time_shape,'i->aibcd')   
        exp_z = csdl.expand(obs_z, time_shape,'i->aibcd')   
        
        exp_r = csdl.expand(radial, time_shape, 'i->abicd')
        
        exp_azim = csdl.expand(azimuth, time_shape, 'i->abcid')
        exp_tau = csdl.expand(tau, time_shape, 'i->abcid')
           
        #====================  observer time computation ======================
        obs_t0 = 30*slope_angle/(np.pi*RPM)   # initial guess for Newton method
        exp_obs_t0 = csdl.expand(obs_t0, time_shape)
        obs_t = csdl.ImplicitVariable(name='obs_t', value = exp_obs_t0.value)   # csdl.nonlinear slover for Newton method
        
        residual_1 = obs_t - ((((exp_x - exp_r*csdl.cos(exp_azim)*csdl.cos(alpha)) + Vinf*(obs_t - exp_tau))**2 + (exp_y - exp_r*csdl.sin(exp_azim)*csdl.cos(alpha))**2 + (exp_z - exp_r*csdl.cos(exp_azim)*csdl.sin(alpha))**2)**(0.5))/c0 - exp_tau
        # residual_2 = 1 - (1/c0)*(((exp_x - exp_r*csdl.cos(exp_azim)*csdl.cos(alpha)) + Vinf*(exp_obs_t - exp_tau))**2 + (exp_y - exp_r*csdl.sin(exp_azim)*csdl.cos(alpha))**2 + (exp_z - exp_r*csdl.cos(exp_azim)*csdl.sin(alpha))**2)**(-0.5)*(exp_x - exp_r*csdl.cos(exp_azim)*csdl.cos(alpha) + Vinf*(exp_obs_t - exp_tau))

        solver = csdl.nonlinear_solvers.Newton('solver_for_observerT')
        solver.add_state(obs_t, residual_1)
        solver.run()
        # # temporary observer time imported from HJ's, due to memory issue
        # obs_t = io.loadmat('OBStime_SUI.mat')
        # obs_t = obs_t['ObsTime']
        # obs_t = csdl.expand(obs_t, time_shape, 'ijk->abjik')
              
        # coordinate transformation
        x_tr = exp_x - exp_r*csdl.cos(exp_azim)*csdl.cos(alpha) + Vinf*(obs_t - exp_tau)
        y_tr = exp_y - exp_r*csdl.sin(exp_azim)*csdl.cos(alpha)
        z_tr = exp_z - exp_r*csdl.cos(exp_azim)*csdl.sin(alpha)
        
        obs_dist_tr = ((x_tr)**2 + (y_tr)**2 + (z_tr)**2)**0.5
        
        # Re-expansion for recovery
        x_tr = csdl.expand(x_tr, target_shape, 'ijkml->ijkmla')
        y_tr = csdl.expand(y_tr, target_shape, 'ijkml->ijkmla')
        z_tr = csdl.expand(z_tr, target_shape, 'ijkml->ijkmla')
        obs_dist_tr = csdl.expand(obs_dist_tr, target_shape, 'ijkml->ijkmla')
        # obs_t = csdl.expand(obs_t, target_shape, 'ijkml->ijkmla')
        
        observer_data_tr = {'num_observers': num_observers,
                            'x_tr': x_tr,
                            'y_tr': y_tr,
                            'z_tr': z_tr,
                            'obs_dist_tr': obs_dist_tr,
                            'obs_time': obs_t
                           }
    
    dir_func = convection_adjustment(observer_data_tr, sectional_mach, machC)

    obs_info = {'S': obs_dist_tr,
                'dh': dir_func
                }
    
    BPMinput = {'a_star': a_star,
                'sectional_mach': sectional_mach,
                'Rc': Rc,
                'u': u,
                'f': f,
                'DT_s': DT_s,
                'DT_p': DT_p, 
                'BT_p': BT_p,
                'l': l,
                'rho': rho,
                'mu': mu,
                'c0': c0
                }
    
    h = BPMVariableGroup.TE_thick
    Psi = BPMVariableGroup.slope_angle
    
    splp, spls, spla, spl_TBLTE, spl_TBLTE_cor = TBLTE(BPMinput, obs_info, num_observers, num_nodes)
    spl_BLUNT = TE_BLUNT(BPMinput, obs_info, num_observers, num_nodes, h, Psi) 
    spl_LBLVS = LBLVS(BPMinput, obs_info, num_observers, num_nodes)    
    
    BPMsum = csdl.power(10, splp/10) + csdl.power(10, spls/10) + csdl.power(10, spla/10) + csdl.power(10, spl_BLUNT/10) # + csdl.power(10, spl_LBLVS) : untripped condition
    totalSPL = 10*csdl.log(BPMsum, 10)  #eq. 20
    # totalSPL0 = totalSPL   #outband conversion check!!
    
    # outband conversion
    if outband == 'one third':
        totalSPL = totalSPL
    else:
        # exp_freq = csdl.expand(f, target_shape, 'i->abcdei')
        narrowSPL = totalSPL - 10.*csdl.log(0.2315*f, 10)
        totalSPL = 10.*csdl.log(int(outband)*csdl.power(10, narrowSPL/10), 10)
        
    SPL_rotor, OASPL = rotor_spl_computation(BPMinput, observer_data_tr, num_nodes, num_radial, num_azim, num_blades, num_freq, totalSPL, flight_condition)
                 
    return SPL_rotor, OASPL
    
    
def convection_adjustment(observer_data, sectional_mach, machC):
    x_r = observer_data['x_tr']
    y_r = observer_data['y_tr']
    z_r = observer_data['z_tr']
    obs_dist = observer_data['obs_dist_tr']
    
    
    x_mag = csdl.arccos(x_r/obs_dist)
    y_z_mag = csdl.arccos(y_r/((y_r**2 + z_r**2))**0.5)
    
    theta = x_mag
    phi = y_z_mag
    dir_func = ((2*csdl.sin(theta/2)**2)*(csdl.sin(phi)**2))/((1+(sectional_mach*csdl.cos(theta)))*(1+(sectional_mach - machC)*csdl.cos(theta))**2)   #EQ B1

    return dir_func     
                
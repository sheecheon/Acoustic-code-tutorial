import numpy as np
import csdl_alpha as csdl


def rotor_spl_computation(BPMinput, observer_data, num_nodes, num_radial, num_azim, num_blades, num_freq, totalSPL, flight_condition):
    
    num_observers = observer_data['num_observers']
    
    if flight_condition == 'hover':
        # ======================= Final Rotor spl computation =========================
        x_tr = observer_data['x_tr']
        S_tr = observer_data['obs_dist_tr']
        
        U = BPMinput['u']
        c0 = BPMinput['c0']
        Mr = U/c0
        
        W = 1 + Mr*(x_tr/S_tr)
                
        # =================== Intial computation for rotor SPL ========================
        Spp_bar = csdl.power(10, totalSPL/10) # note: shape is (num_nodes, num_observers, num_radial, num_azim, num_freq)
        Spp_func = (2*np.pi/(num_azim-1))*(W**2)*Spp_bar      # Spp_func = (W**2)*Spp_bar  # ok
        Spp_R = num_blades*(1/(2*np.pi))*csdl.sum(Spp_func, axes=(3,))
        Spp_rotor = csdl.sum(Spp_R, axes=(2,))
        
        SPL_rotor = 10*csdl.log(Spp_rotor, 10)
        OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)
        
        return SPL_rotor, OASPL
        
        
    elif flight_condition == 'edgewise':
        # ================= Find time index according to time frame ===================
        obs_time = observer_data['obs_time']
        tMin = csdl.minimum(obs_time, rho = 1000000.).value
        tMax = csdl.maximum(obs_time, rho = 1000000.).value
        
        num_tRange = 40  # # of time step is arbitrary chosen!! 
        tRange = np.linspace(tMin, tMax, num_tRange)
        dt = tRange[2] - tRange[1]
        
        time_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_tRange)
        exp_tRange = csdl.expand(csdl.reshape(tRange, (num_tRange,)), time_shape, 'i->abcdei')
        exp2_obs_time = csdl.expand(obs_time, time_shape, 'ijklm->ijklma')
        
        time_dist = ((exp_tRange - exp2_obs_time)**2)**0.5
        arr_time_dist = time_dist.value
        
        sorted_dist = np.argsort(arr_time_dist, axis = -1)  
        time_indx1 = sorted_dist[:, :, :, :, :, 0]    # smallest dist index (Idx) 
        time_indx2 = sorted_dist[:, :, :, :, :, 1]    # 2nd smallest dist index (Idx+1 or Idx-1)
        
        min_tRange1  = np.take(exp_tRange.value, time_indx1) # Select appropriate time value to be allocated
        min_tRange2 = np.take(exp_tRange.value, time_indx2)  # Select appropriate time value 2 to be allocated
        
        target_shape = (num_nodes, num_observers, num_radial, num_azim, num_blades, num_freq)
        time_coeff1 = csdl.expand((((min_tRange1 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')
        time_coeff2 = csdl.expand((((min_tRange2 - obs_time)**2)**0.5)/dt, target_shape, 'ijklm->ijklma')
        
        # Compute noise contribution
        noise_con1 = time_coeff1*csdl.power(10, totalSPL/10)
        noise_con2 = time_coeff2*csdl.power(10, totalSPL/10)
        
        # ================ Allocate noise contribution to time frame ==================
        # start = time.time()
        time_indx1 = csdl.Variable(value = time_indx1)
        time_indx2 = csdl.Variable(value = time_indx2)
        sumSPL = csdl.Variable(shape=(num_nodes, num_observers, num_tRange, num_freq), value=0)
        for k in csdl.frange(num_blades):
            for j in csdl.frange(num_azim): 
                for i in csdl.frange(num_radial):
                    closestIndx = time_indx1[:, :, i, j, k]
                    closestIndx2 = time_indx2[:, :, i, j, k]
                    sumSPL = sumSPL.set(csdl.slice[:, :, closestIndx, :], sumSPL[:, :, closestIndx, :] + noise_con1[:, :, i, j, k, :])
                    sumSPL = sumSPL.set(csdl.slice[:, :, closestIndx2, :], sumSPL[:, :, closestIndx2, :] + noise_con2[:, :, i, j, k, :])
        # end = time.time()
        # print('time consuming for HJ loops :', end - start)
        
        SPL_rotor0 = 10*csdl.log(sumSPL, 10)
        SPL_rotor = csdl.sum(SPL_rotor0, axes=(2,))/num_tRange
        OASPL = 10*csdl.log(csdl.sum(csdl.power(10, SPL_rotor/10)), 10)

        return SPL_rotor, OASPL
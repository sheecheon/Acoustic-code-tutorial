# obs_tutorial_SC : 'steady observer_model' as fuction
import numpy as np
from lsdo_acoustics.core.acoustics import Acoustics
import csdl_alpha as csdl 

# a = Acoustics(aircraft_position=np.array([0., 0., 0.,]))
# num_nodes = 1

# a.add_observer(
#     name='observer',
#     obs_position=np.array([1.859475, 0, -1.302018]),
#     time_vector=np.array([0.,]).reshape((1,))
# )

# observer_data = a.assemble_observers()
# velocity_data = np.array([0.,0.,0.])  # Q1 : steady -> velocity = 0 ?
# # velocity_data = np.array([0,0,0])


# observer_data['name']

# RPM = 5535.0
# # num_nodes = 1

# # Q : without recorder inside of fuction file, 
recorder = csdl.Recorder(inline=True)
recorder.start()

def steady_observer_model(observer_data, velocity_data, num_nodes = 1):
    # num_nodes = observer_data['num_nodes']
    aircraft_location = observer_data['aircraft_position']
    x = observer_data['x']
    y = observer_data['y']
    z = observer_data['z']
    
    time_vectors = observer_data['time']
    num_obs = observer_data['num_observers']

    obs_x_loc = csdl.Variable(value=x)
    obs_y_loc = csdl.Variable(value=y)
    obs_z_loc = csdl.Variable(value=z)
    

    Vx = csdl.Variable(shape = (num_nodes,), value = 0.)
    Vy = csdl.Variable(shape = (num_nodes,), value = 0.)
    Vz = csdl.Variable(shape = (num_nodes,), value = 0.)
    # Vx = np.array([0.])


    V_aircraft = csdl.Variable(shape=(num_nodes,3), value = 0.)
    V_aircraft = V_aircraft.set(csdl.slice[:,0], value=Vx)
    V_aircraft = V_aircraft.set(csdl.slice[:,1], value=Vy)
    V_aircraft = V_aircraft.set(csdl.slice[:,2], value=Vz)

    exp_shape = (num_nodes, 3, num_obs)
    V_exp = csdl.expand(V_aircraft, exp_shape, action='ij->ija')
    
    if num_obs == 1:
        time = csdl.expand(time_vectors, exp_shape)
    else:
        time = csdl.expand(time_vectors, exp_shape, action='i->abi')

    aircraft_location = csdl.expand(aircraft_location, exp_shape, action='ij->aij')
    aircraft_x_pos = aircraft_location[:,0,:] + V_exp[:,0,:]*time[:,0,:]
    aircraft_y_pos = aircraft_location[:,1,:] + V_exp[:,1,:]*time[:,1,:]
    aircraft_z_pos = aircraft_location[:,2,:] + V_exp[:,2,:]*time[:,2,:]

    '''
    OUTPUTS:
    - obs location relative to aircraft
    '''
    rel_obs_x_pos = obs_x_loc - aircraft_x_pos
    rel_obs_y_pos = obs_y_loc - aircraft_y_pos
    rel_obs_z_pos = obs_z_loc - aircraft_z_pos
    
    #
    rel_obs_position = csdl.Variable(shape = exp_shape, value = 0.)
    rel_obs_position = rel_obs_position.set(csdl.slice[:,0,:], value = rel_obs_x_pos)
    rel_obs_position = rel_obs_position.set(csdl.slice[:,1,:], value = rel_obs_y_pos)
    rel_obs_position = rel_obs_position.set(csdl.slice[:,2,:], value = rel_obs_z_pos)

    
    rel_obs_dist = (rel_obs_x_pos**2 + rel_obs_y_pos**2 + rel_obs_z_pos**2)**(0.5)
    
    
    # Thrust direction and angles
    theta = csdl.Variable(shape = (num_nodes,1), value = 0.*np.pi/180.)
    rotation_matrix = csdl.Variable(shape = (3,3), value = 0.) 
    # ONLY CONSIDERING PITCH CHANGES (X-Z), NO YAW OR ROLL FOR NOW
    rotation_matrix = rotation_matrix.set(csdl.slice[1,1], value = (theta + 10)/(theta + 10))
    rotation_matrix = rotation_matrix.set(csdl.slice[0,0], value = csdl.cos(theta))
    rotation_matrix = rotation_matrix.set(csdl.slice[0,2], value = -1 * csdl.sin(theta))
    rotation_matrix = rotation_matrix.set(csdl.slice[2,0], value = -1 * csdl.sin(theta))
    rotation_matrix = rotation_matrix.set(csdl.slice[2,2], value = -1 * csdl.cos(theta))

    thrust_vec = csdl.Variable(shape=(3, ), value = np.array([0., 0., 1.])) # Note1: 'thrust_vec' input temporary
    thrust_dir = csdl.matvec(rotation_matrix, thrust_vec/csdl.expand(csdl.norm(thrust_vec), out_shape=(3,)))
    thrust_dir = csdl.expand(thrust_dir, exp_shape, action='i->aib') #should be removed 

    normal_proj = csdl.sum(csdl.product(rel_obs_position, thrust_dir), axes=(1,))

    # normal_proj = csdl.tensordot(rel_obs_position, thrust_dir, axes = ([1,0],[1,0]))  # Note 2: How choose 'tensordot' option (axis)
    #asdf

    # rel_angle_plane = csdl.arcsin(csdl.expand(normal_proj, (num_nodes, 1, num_obs))/rel_obs_dist) 
    # rel_angle_normal = csdl.arccos(csdl.expand(normal_proj, (num_nodes, 1, num_obs))/rel_obs_dist) 
        
    rel_angle_plane = csdl.arcsin(csdl.expand(normal_proj, (num_nodes, 1, num_obs), action='ij->ija')/rel_obs_dist) 
    rel_angle_normal = csdl.arccos(csdl.expand(normal_proj, (num_nodes, 1, num_obs), action='ij->ija')/rel_obs_dist) 
    
    rel_angle_plane = csdl.reshape(rel_angle_plane, (num_nodes, num_obs))
    rel_angle_normal = csdl.reshape(rel_angle_normal, (num_nodes, num_obs))
    
    # rel_obs_angle = csdl.arccos(rel_obs_z_pos / rel_obs_dist) *  \
    #                (rel_obs_z_pos + 1e-12) / ((rel_obs_z_pos + 1e-12)**2)**(0.5)

    return rel_obs_dist, rel_angle_plane, rel_angle_normal

# recorder = csdl.Recorder(inline=True)
# recorder.start()
# rel_obs_dist, rel_angle_plane, rel_angle_normal = steady_observer_model(observer_data, velocity_data)
# recorder.stop()
# print('real_obas_dist', rel_obs_dist.value)
# print('rel_angle_plane', rel_angle_plane.value)
# print('rel_angle_normal', rel_angle_normal.value)
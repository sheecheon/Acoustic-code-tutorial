import numpy as np
# import csdl_alpha as csdl
import matplotlib.pyplot as plt

# recorder = csdl.Recorder(inline = True)
# recorder.start()

freq = np.array([400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                 6300,8000,10000,12500,16000,20000,25000,31500,40000,50000])  # 1/3 octave band central frequency [Hz]

# BPM_HJ_SPL_rotor = np.array([53.84561889, 51.93408201, 51.01285792, 51.22285315,
#                              52.05486659, 53.19403159, 54.50994335, 54.71046721,
#                              54.26593135, 53.44549374, 51.83080989, 49.77820022,
#                              47.15846616, 44.40768276, 42.30951345, 40.62819023,
#                              35.96992061, 30.14233348, 25.43983054, 22.64790755,
#                              20.12249521, 17.33687391])

# BPM_SH_initial = np.array([54.0879, 51.8509, 50.1069, 49.6171, 50.6176,
#                            52.1011, 53.2565, 53.5193, 52.9675, 51.6861,
#                            49.7011, 47.0361, 44.0142, 41.1358, 40.0094,
#                            37.6205, 32.1989, 28.3011, 26.0512, 24.8206,
#                            22.5699, 19.9934])

# case1 = np.array([52.01648623, 49.53533866, 47.35998011, 45.79845302,
#                   45.48843183, 46.05780963, 46.52981858, 46.32986494,
#                   45.5525682 , 44.16247593, 41.69580578, 38.22997865,
#                   34.42034765, 31.22069756, 28.67549491, 26.27913184,
#                   23.46815703, 21.74485341, 19.3742892 , 15.41257679,
#                   12.40628518, 9.379443793])

# case2  = np.array([57.05837667, 54.06118811, 51.28557673, 49.1878447 ,
#                    48.9162938 , 50.16425062, 51.5336138 , 51.86148455,
#                    51.12870966, 49.31007509, 46.44252333, 42.83526253,
#                    39.12177401, 36.77364955, 35.82923983, 30.82568925,
#                    22.58798656, 21.63487701, 21.54929216, 20.92356084,
#                    18.50642555, 15.03428782])

# case4  = np.array([57.42413675, 54.61304806, 51.81321761, 49.43199153, 
#                    48.44863659, 48.70987849, 49.07604993, 48.72971678, 
#                    47.50600966, 45.21200633, 41.45119722, 37.54212405,
#                    33.06083176, 29.91816143, 28.23655865, 25.68781415,
#                    24.99068308, 23.93437821, 20.31505338, 16.8599918 ,
#                    14.73628537, 12.36544934])

# OASPL = 10*csdl.log(csdl.sum(csdl.power(10, BPM_SH_initial/10)), 10)
# OASPL1 = 10*csdl.log(csdl.sum(csdl.power(10, case1/10)), 10)
# OASPL2 = 10*csdl.log(csdl.sum(csdl.power(10, case2/10)), 10)
# OASPL4 = 10*csdl.log(csdl.sum(csdl.power(10, case4/10)), 10)

# print('OASPL :', OASPL.value)
# print('OASPL1 :',OASPL1.value)
# print('OASPL2 :',OASPL2.value)
# print('OASPL4 :',OASPL4.value)

# rel_error = abs((BPM_HJ_SPL_rotor - BPM_SH_initial))/BPM_HJ_SPL_rotor
# rel_error1 = abs((BPM_HJ_SPL_rotor - case1))/BPM_HJ_SPL_rotor
# rel_error2 = abs((BPM_HJ_SPL_rotor - case2))/BPM_HJ_SPL_rotor
# rel_error4 = abs((BPM_HJ_SPL_rotor - case4))/BPM_HJ_SPL_rotor

# print('Relative erorr initial :', rel_error)
# print('Relative erorr case1 :', rel_error1)
# print('Relative erorr case2 :', rel_error2)
# print('Relative erorr case4 :', rel_error4)

# plt.figure()
# plt.semilogy(freq, rel_error) 
# plt.semilogy(freq, rel_error1)
# plt.semilogy(freq, rel_error2) 
# plt.semilogy(freq, rel_error4) 

# plt.legend(['Intial Relative error', 'Case1', 'Case2', 'Case4'])
# plt.title('Relative error of BPM model Validation for freestream velocity')
# plt.xlabel('Frequency [HZ]')
# plt.ylabel('Relative error')
# plt.grid()
# plt.show()

# ##### Re-plot

# case1 = np.array([24.43114069, 22.6371519, 20.68215334, 18.75741163,
#                   17.05301603, 15.21255148, 12.89850278, 10.63105959,
#                   7.720259469, 3.996616201, -0.650130066, -3.438934085,
#                   -5.975069736, -8.183644274, -9.145054299, -10.9596585,
#                   -13.06234347, -15.60218224, -19.16794669, -23.63542675,
#                   -30.41913428, -38.07111967])

# case2 = np.array([76.93164382, 73.78068307, 70.74194143, 68.31126578,
#                   67.82333528, 69.22929155, 71.19644619, 72.31772746,
#                   72.69340128, 72.18839941, 70.76800721, 68.7079796,
#                   65.70997329, 61.89611137, 58.19698822, 55.29349796,
#                   53.95841715, 54.34648085, 52.93284341, 46.15430041,
#                   40.67745625, 38.53590931])

# plt.figure()
# plt.plot(freq, BPM_HJ_SPL_rotor)
# plt.plot(freq, case1) 
# plt.plot(freq, case2)
# plt.legend(['SPL-HJ', 'Case1', 'Case2', 'Case4'])
# plt.xlabel('Frequency [HZ]')
# plt.ylabel('SPL [dB]')
# plt.grid()
# plt.show()


## 
# import matplotlib.pyplot as plt
rel_error = np.array([0.01328785, 0.02002385, 0.03516478, 0.04572587, 0.03642444,
                      0.02334319, 0.02300203, 0.02499928, 0.02661258, 0.03361992,
                      0.04219295, 0.05358436, 0.07342147, 0.07928446, 0.05958261,
                      0.07500692, 0.10580046, 0.06365608, 0.00778027, 0.07428821,
                      0.10438317, 0.13572187])

rel_error2 = np.array([0.01328785, 0.02002385, 0.03516478, 0.04572587, 0.03642444,
                      0.02334319, 0.02300203, 0.02499928, 0.02661258, 0.03361992,
                      0.04219295, 0.05358436, 0.07342147, 0.07928446, 0.05958261,
                      0.07500692, 0.10580046, 0.06365608, 0.00778027, 0.07428821,
                      0.10438317, 0.13572187])



plt.figure()
plt.semilogy(freq, rel_error, label="V0 : from HJ data")
plt.semilogy(freq, rel_error2, label="V0 : obtained by tangential velocitys")
plt.xlabel('Frequency [HZ]')
plt.ylabel('Relative error')
plt.legend(['V0 : from HJ data', 'V0 : obtained by tangential velocity'], loc=1)
plt.grid()
plt.show()

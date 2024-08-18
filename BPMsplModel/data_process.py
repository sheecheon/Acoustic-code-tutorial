# Data import or save to pkl

from scipy import io
import pickle


temp_input = io.loadmat('HoverInput_S76.mat')

HoverInput_S76 = temp_input

# 파일로 저장
with open('HoverInput_S76', 'wb') as f:
    pickle.dump(HoverInput_S76, f)
    
# 파일 불러오기
# with open('my_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)


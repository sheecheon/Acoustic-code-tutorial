# Data import or save to pkl

from scipy import io
import pickle


temp_input = io.loadmat('EdgewiseInput_BO105Case1.mat')

EdgewiseInput_BO105Case1 = temp_input

# 파일로 저장
with open('EdgewiseInput_BO105Case1', 'wb') as f:
    pickle.dump(EdgewiseInput_BO105Case1, f)
    
# 파일 불러오기
# with open('my_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)


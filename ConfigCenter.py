from datetime import datetime
import os
import tensorflow as tf
from module_garden.company_merge import read_company_list
from module_garden.position_merge import read_position_list
from module_garden.time_next import next_month
from root_path import root


# sy_position
class GlobalConfig_sy():
    def __init__(self):
        # self.MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.MATRIX_ROOT=r'../Data/zl_filter/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/cp_filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.TRAIN_MATRIX_ROOT=r'../Data/zl_filter/matrix_dict_'

        self.VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model'
        self.SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com-2/model'

        # self.NN_DIR=r'../Data/NN_Model/ver1-32-2-simple_2/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-3level-7/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-side_lstm-cross_new_loss-3/model'
        self.NN_DIR = r'../Data/NN_Model/nn-MHA-side-cross-3/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-1/model'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/ver1-32-2-simple_2/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-3level-7/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-side_lstm-cross_new_loss-2/result_matrix_'
        self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-side-cross-3/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-1/result_matrix_'
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])

        self.LOAD_SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com/model-40000'
        self.LOAD_VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model-94001'

        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39
        # self.dimension = 10
        # self.dimension = 32
        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        # self.past_window=12
        # self.future_window=6

        self.dimension = 60

        # Transformer Parameter
        self.d_original = self.dimension
        self.n_head = 8
        self.d_head = 32
        self.d_inner = 256
        self.m_dense = 3
        self.d_spcific_inner = 128
        self.d_side_inner = 128
        self.d_side_out = 8
        # self.d_model=None
        self.d_model = 64
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = True
        self.clip = 0.25
        self.shared_num = 3
        self.specific_num = 1
        self.regularization_decay = 0.001
        self.mtl_regularization_decay = 0.0001
        self.m_loss_decay = 1 / 100
        self.lstm_len = 4
        self.lstm_unit = 60
        self.lstm_layers = 2

        self.pre_learning_rate = 0.1
        self.pre_len_time = 12
        self.pre_len_company = 8
        self.pre_len_positon = self.position_num

        self.single_learning_rate = 0.1
        self.single_len_time = 1
        self.single_len_company = self.company_num
        self.single_len_positon = self.position_num

        self.nn_loop = 10

        self.nn_learning_rate = 0.0002
        self.nn_past_len_time = 12
        # self.nn_past_len_time = 8
        self.nn_future_len_time = 12
        # self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        self.nn_train_len_company = 16
        self.nn_test_len_company = self.company_num
        self.nn_len_positon = self.position_num
        # self.nn_side_len=66
        self.nn_side_len = 76

class GlobalConfig_zl_0():
    def __init__(self):
        # self.MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.MATRIX_ROOT=r'../Data/zl_filter/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/cp_filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.TRAIN_MATRIX_ROOT=r'../Data/zl_filter/matrix_dict_'

        self.VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model'
        self.SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com-2/model'

        # self.NN_DIR=r'../Data/NN_Model/ver1-32-2-simple_2/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-3level-7/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-side_lstm-cross_new_loss-3/model'
        self.NN_DIR = r'../Data/NN_Model/nn-MHA-side-cross-zl_6/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-1/model'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/ver1-32-2-simple_2/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-3level-7/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-side_lstm-cross_new_loss-2/result_matrix_'
        self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-side-cross-zl_6/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-1/result_matrix_'
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])

        self.LOAD_SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com/model-40000'
        self.LOAD_VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model-94001'

        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39
        # self.dimension = 10
        # self.dimension = 32
        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        # self.past_window=12
        # self.future_window=6

        self.dimension = 17

        # Transformer Parameter
        self.d_original = self.dimension
        self.n_head = 8
        self.d_head = 32
        self.d_inner = 256
        self.m_dense = 3
        self.d_spcific_inner = 128
        self.d_side_inner = 128
        self.d_side_out = 8
        # self.d_model=None
        self.d_model = 32
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = True
        self.clip = 0.25
        self.shared_num = 2
        self.specific_num = 1
        self.regularization_decay = 0.001
        self.mtl_regularization_decay = 0.0001
        self.m_loss_decay = 1 / 152
        self.lstm_len = 4
        self.lstm_unit = 16
        self.lstm_layers = 2

        self.pre_learning_rate = 0.1
        self.pre_len_time = 12
        self.pre_len_company = 8
        self.pre_len_positon = self.position_num

        self.single_learning_rate = 0.1
        self.single_len_time = 1
        self.single_len_company = self.company_num
        self.single_len_positon = self.position_num

        self.nn_loop = 10

        self.nn_learning_rate = 0.0002
        self.nn_past_len_time = 12
        # self.nn_past_len_time = 8
        self.nn_future_len_time = 12
        # self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        self.nn_train_len_company = 16
        self.nn_test_len_company = self.company_num
        self.nn_len_positon = self.position_num
        # self.nn_side_len=66
        self.nn_side_len = 76

class GlobalConfig():
    def __init__(self):
        # self.MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.MATRIX_ROOT=r'../Data/zl16_filter_stdnorm/matrix_dict_'
        self.MATRIX_ROOT_test=r'../Data/zl16_test/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/zl16_filter_stdnorm_cp/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/cp_filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/sy_select_filter/matrix_dict_'
        self.TRAIN_MATRIX_ROOT=r'../Data/zl16_filter_stdnorm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/zl16_filter_stdnorm_cp/matrix_dict_'

        self.VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model'
        self.SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com-2/model'

        # self.NN_DIR=r'../Data/NN_Model/ver1-32-2-simple_2/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-3level-7/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-side_lstm-cross_new_loss-3/model'
        # self.NN_DIR = r'../Data/NN_Model/nn-MHA-side-cross-zl_7/model'
        self.NN_DIR = r'../Data/NN_Model/nn-MHA-SE-zl16/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-1/model'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/ver1-32-2-simple_2/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-3level-7/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-side_lstm-cross_new_loss-2/result_matrix_'
        self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-SE-zl16/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-1/result_matrix_'
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])

        self.LOAD_SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com/model-40000'
        self.LOAD_VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model-94001'

        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39
        # self.dimension = 10
        # self.dimension = 32
        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        # self.past_window=12
        # self.future_window=6

        self.dimension = 16

        # Transformer Parameter
        self.d_original = self.dimension
        # self.n_head = 8
        # self.d_head = 32
        # self.d_inner = 256
        # self.m_dense = 3
        # self.d_spcific_inner = 64
        self.n_head = 16
        self.d_head = 8
        self.d_inner = 128
        self.m_dense = 3
        self.d_spcific_inner = 64

        self.d_side_inner = 128
        self.d_side_out = 4
        # self.d_model=None
        self.d_model = 16
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = False
        self.clip = 0.25
        self.shared_num = 3
        self.specific_num = 1
        self.regularization_decay = 0.001
        self.mtl_regularization_decay = 0.0001
        self.m_loss_decay = 1 / 152
        self.lstm_len = 4
        self.lstm_unit = 16
        self.lstm_layers = 2

        self.pre_learning_rate = 0.1
        self.pre_len_time = 12
        self.pre_len_company = 8
        self.pre_len_positon = self.position_num

        self.single_learning_rate = 0.1
        self.single_len_time = 1
        self.single_len_company = self.company_num
        self.single_len_positon = self.position_num

        self.nn_loop = 10

        self.nn_learning_rate = 0.0002
        self.nn_past_len_time = 12
        # self.nn_past_len_time = 8
        self.nn_future_len_time = 12
        # self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        self.nn_train_len_company = 16
        self.nn_test_len_company = self.company_num
        self.nn_len_positon = self.position_num
        # self.nn_side_len=66
        self.nn_side_len = 76

class GlobalConfig_CP_std():
    def __init__(self):
        self.MATRIX_ROOT=r'../Data/zl16_filter_stdnorm/matrix_dict_'
        self.TRAIN_MATRIX_ROOT=r'../Data/zl16_filter_stdnorm/matrix_dict_'

        # self.NN_DIR = r'../Data/NN_Model/nn-MHA-CP-zl16-4/model'
        self.NN_DIR = r'../Data/NN_Model/lstm1/model'
        # self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-CP-zl16-4/result_matrix_'
        self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/lstm1/result_matrix_'
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])


        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39

        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        self.d_step=4
        self.dimension = 16

        # Transformer Parameter
        # self.classify_num=5
        self.classify_num=5
        self.d_original = self.dimension
        # self.n_head = 8
        # self.d_head = 32
        # self.d_inner = 256
        # self.m_dense = 3
        # self.d_spcific_inner = 64
        self.n_head = 8
        self.d_head = 8
        self.d_inner = 128
        self.d_input_inner=32
        self.m_dense = 3
        self.d_spcific_inner = 64

        self.d_side_inner = 32
        self.d_side_out = 4
        # self.d_model=None
        self.d_model = 16
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = True
        self.clip = 0.25
        self.shared_num = 3
        self.specific_num = 1
        # self.regularization_decay = 0.001
        self.regularization_decay = 0.0001
        # self.regularization_decay = 0
        # self.mtl_regularization_decay = 0.0001
        self.mtl_regularization_decay = 0.0001
        # self.mtl_regularization_decay = 0
        # self.m_loss_decay = 1 / 152
        self.lstm_len = 4
        self.lstm_unit = 16
        self.lstm_layers = 2

        self.nn_loop = 380

        #'r'是回归label，'s'是sax离散化label，'u'是反应相比于上个月上升还是下降的离散化label
        # self.label_type='s'
        self.label_type='u'
        # self.test_num=12
        self.test_num=12
        self.batch_size=32
        self.test_batch_size=12
        self.nn_learning_rate = 0.0002
        # self.nn_learning_rate = 0.01
        self.nn_past_len_time = 12
        # self.nn_past_len_time = 8
        self.nn_future_len_time = 12
        # self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        # self.nn_train_len_company = 16
        # self.nn_test_len_company = self.company_num
        # self.nn_len_positon = self.position_num
        # self.nn_side_len=66
        self.nn_side_len = 82

class GlobalConfig_CP():
    def __init__(self):
        self.MATRIX_ROOT=os.path.join(root, r'Data/zl16_filter_stdnorm/matrix_dict_')
        self.TRAIN_MATRIX_ROOT=os.path.join(root, r'Data/zl16_filter_stdnorm/matrix_dict_')

        # self.NN_DIR = r'../Data/NN_Model/nn-MHA-CP-zl16-4/model'
        # save_name = 'WWW_find_base_925-3'
        # save_name = 'WWW_find_baseTrans_1017-2'
        save_name = 'KDD_baseLSTnet_205-3'
        # save_name = 'WWW_find_base_914-5'
        reload_name = 'WWW_find_base_921-4'
        self.NN_DIR = os.path.join(root, r"Data/NN_Model/" + save_name + "/model")
        self.NN_DIR_win = os.path.join(root, r"Data/NN_Model/" + save_name + "/model_win")
        # self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-CP-zl16-4/result_matrix_'
        self.TEST_RESULT_DIR = os.path.join(root, r"Data/NN_Model_Test_Result/" + save_name + "/result_matrix_")
        # self.reload_path= os.path.join(root, r"Data/NN_Model/" + reload_name + "/model_win-1980")
        # self.reload_path= os.path.join(root, r"Data/NN_Model/" + reload_name + "/model_win-40")
        self.reload_path= False
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])
        if os.path.exists(os.path.split(self.NN_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.NN_DIR)[0])
        if os.path.exists(os.path.split(self.NN_DIR_win)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.NN_DIR_win)[0])

        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39

        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        self.d_step=1
        self.dimension = 16

        # Transformer Parameter
        self.classify_num=5
        # self.classify_num=3
        self.d_original = self.dimension
        # self.n_head = 8
        # self.d_head = 32
        # self.d_inner = 256
        # self.m_dense = 3
        # self.d_spcific_inner = 64
        self.n_head = 4
        self.d_head = 4
        self.d_inner = 32
        self.d_input_inner=32
        self.m_dense = 3
        self.d_spcific_inner = 32

        # self.d_model=None
        self.d_model = 16
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = True
        self.clip = 0.25
        self.shared_num = 3
        self.specific_num = 1
        # self.regularization_decay = 0.0005
        self.regularization_decay = 0.005
        # self.regularization_decay = 0.0005
        # self.regularization_decay = 0.0001
        # self.regularization_decay = 0
        # self.mtl_regularization_decay = 0.0001
        self.mtl_regularization_decay = 0.0001
        # self.mtl_regularization_decay = 0
        # self.m_loss_decay = 1 / 152
        self.lstm_len = 4
        self.lstm_unit = 16
        self.lstm_layers = 2

        self.nn_loop = 380
        # self.nn_loop = 380*4

        #'r'是回归label，'s'是sax离散化label，'u'是反应相比于上个月上升还是下降的离散化label
        # self.label_type='s'
        self.label_type='u'
        # self.test_num=12
        self.test_num=12
        self.batch_size=32
        self.test_batch_size=1
        # self.nn_learning_rate = 0.0002
        self.nn_learning_rate = 0.002
        # self.nn_learning_rate = 0.0001
        # self.nn_learning_rate = 0.01
        # self.nn_past_len_time = 4
        self.nn_past_len_time =6
        # self.nn_past_len_time = 8
        # self.nn_future_len_time = 4
        self.nn_future_len_time = 6
        # self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        # self.nn_train_len_company = 16
        # self.nn_test_len_company = self.company_num
        # self.nn_len_positon = self.position_num
        # self.nn_side_len=66
        self.nn_side_len = 82
        self.d_side_inner = 64
        self.d_side_out = self.n_head*self.d_head*self.nn_past_len_time*2


# 使用原始为筛选数据，职位较少
class GlobalConfig_v1():
    def __init__(self):
        self.MATRIX_ROOT=r'../Data/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/sy_com/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'
        # self.MATRIX_ROOT = r'../Data/cp_filter_and_norm/matrix_dict_'
        self.TRAIN_MATRIX_ROOT = r'../Data/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT = r'../Data/sy_com/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_norm/matrix_dict_'
        # self.TRAIN_MATRIX_ROOT=r'../Data/filter_and_no_norm/matrix_dict_'

        self.VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model'
        self.SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com-2/model'

        # self.NN_DIR=r'../Data/NN_Model/ver1-32-2-simple_2/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-3level-7/model'
        # self.NN_DIR = r'../Data/NN_Model/nn-MHA-lstm-cross-1/model'
        self.NN_DIR = r'../Data/NN_Model/test-1/model'
        # self.NN_DIR=r'../Data/NN_Model/nn-MHA-1/model'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/ver1-32-2-simple_2/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-3level-7/result_matrix_'
        # self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/nn-MHA-lstm-cross-1/result_matrix_'
        self.TEST_RESULT_DIR = r'../Data/NN_Model_Test_Result/test-1/result_matrix_'
        # self.TEST_RESULT_DIR=r'../Data/NN_Model_Test_Result/nn-MHA-1/result_matrix_'
        if os.path.exists(os.path.split(self.TEST_RESULT_DIR)[0]):
            pass
        else:
            os.makedirs(os.path.split(self.TEST_RESULT_DIR)[0])

        self.LOAD_SINGLE_AND_VECTOR_DIR = r'../Data/Single_and_Vector_Model/ver1-32-2-all-com/model-40000'
        self.LOAD_VECTOR_DIR = r'../Data/Vector_Model/ver1-32-2/model-94001'

        self.company_list = read_company_list()
        self.position_list = read_position_list()
        self.company_num = len(self.company_list)
        self.position_num = len(self.position_list)

        self.starttime = 201506
        self.endtime = 201808
        self.time_list = [self.starttime]
        time_one = self.starttime
        while int(time_one) < int(self.endtime):
            time_one = next_month(time_one, 1)
            self.time_list.append(time_one)
        self.all_time_length = 39
        # self.dimension = 10
        # self.dimension = 32
        # Parameter initialization
        self.D_TYPE = tf.float32
        self.Stddev = 0.2
        self.INITIALIZER = tf.initializers.random_normal(stddev=0.2, seed=None)
        # self.INITIALIZER=tf.initializers.random_uniform(minval=-0.1,maxval=0.1,seed=None)

        # self.past_window=12
        # self.future_window=6

        self.dimension = 32

        # Transformer Parameter
        self.d_original = self.dimension
        self.n_head = 16
        self.d_head = 16
        self.d_inner = 256
        self.m_dense = 3
        self.d_spcific_inner = 128
        self.d_side_inner = 128
        self.d_side_out = 8
        # self.d_model=None
        self.d_model = 64
        self.only_last_one = True
        self.test_all = False
        self.dropout = 0.1
        self.dropatt = 0
        self.mask_same_length = False
        self.init_embeding = True
        self.clip = 0.25
        self.shared_num = 3
        self.specific_num = 1
        self.regularization_decay = 0.001
        self.mtl_regularization_decay = 0.0001
        self.m_loss_decay = 1 / 8
        self.lstm_len = 4
        self.lstm_unit = 33
        self.lstm_layers = 2

        self.pre_learning_rate = 0.1
        self.pre_len_time = 12
        self.pre_len_company = 8
        self.pre_len_positon = self.position_num

        self.single_learning_rate = 0.1
        self.single_len_time = 1
        self.single_len_company = self.company_num
        self.single_len_positon = self.position_num

        self.nn_loop = 10

        self.nn_learning_rate = 0.0002
        # self.nn_past_len_time = 12
        self.nn_past_len_time = 8
        # self.nn_future_len_time = 12
        self.nn_future_len_time = 8
        # 维持一致（lstm模块要求）
        # self.nn_train_len_company = self.company_num
        self.nn_train_len_company = 32
        self.nn_test_len_company = self.company_num
        self.nn_len_positon = self.position_num
        self.nn_side_len = 76

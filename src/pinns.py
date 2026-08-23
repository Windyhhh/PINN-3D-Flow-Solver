
import numpy as np
from userbackend import tf, _GPU_NUM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras import backend
from maps import generator
from custom_lbfgs import lbfgs, Struct
import os
import scipy.io as sio
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize 



"""
user defined informantion print
"""

class LossPrintingCallback(Callback):                                  # 将loss打印回弹显示在工作区
    def __init__(self, pinn_model, uvw_log_path, sample_num=2000):     # 将每个轮次抽样点的最大最小值写入txt文件
        super(LossPrintingCallback, self).__init__()
        self.pinn_model = pinn_model
        self.uvw_log_path = uvw_log_path
        self.sample_num = sample_num

        # 每次新训练开始时清空文件并写表头
        with open(self.uvw_log_path, "w") as f:
            f.write("epoch, u_min, u_max, v_min, v_max, w_min, w_max\n")

    def on_epoch_end(self, epoch, logs=None):
        # Keras 3 使用 learning_rate 而不是 lr
        lr = float(backend.get_value(self.model.optimizer.learning_rate))
        print("Epoch %05d: loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e, learning rate: %8.6f" %
                (epoch, logs["loss"], logs["loss_fn_data"], logs["loss_fn_eqns"], logs["loss_fn_conds"], lr)
        )

        # ======== 从 train_X中随机随机 N 个采样点 ========
        X_all = self.pinn_model.train_X
        N_all = X_all.shape[0]
        N_sample = min(self.sample_num, N_all)  
        idx = np.random.choice(N_all, N_sample, replace=False)
        X_sample = X_all[idx, :]

        # 只对这 N_sample 个点做一次前向预测，统计其对应的预测值的最大最小值
        u, v, w, _ = self.pinn_model.get_uvwp(X_sample)

        u_min, u_max = float(tf.reduce_min(u)), float(tf.reduce_max(u))
        v_min, v_max = float(tf.reduce_min(v)), float(tf.reduce_max(v))
        w_min, w_max = float(tf.reduce_min(w)), float(tf.reduce_max(w))

        with open(self.uvw_log_path, "a") as f:
            f.write(f"{epoch}, {u_min:.6e}, {u_max:.6e}, "
                    f"{v_min:.6e}, {v_max:.6e}, "
                    f"{w_min:.6e}, {w_max:.6e}\n")


    
        

class NavierStokes3DPINNs(object):
    def __init__(self, hp, data, eqns, *conds):                        # 由datagenerator.py生成的数据，main程序设置的超参数
        """                                                            
        data points:    [t,x,y,z,u,v,w,p]
        eqns points:    [t,x,y,z,0,0,0,0]
        conds:          other conditions like boundaries      
        Note: equation points should include the points of boundary conditions
        """
        # clear session
        tf.keras.backend.clear_session()
        
        # hp is the structure of hyper-parameters
        self.dtype = 'float32'
        self.layers = hp['layers']
        self.ExistModel = hp['ExistModel']
        self.map_name = hp['map_name']
        self.savename = hp['savename']
        self.Re = hp['Re']
        self.alpha = hp['alpha']   
        self.z_weight = hp['z_weight']  
        self.dim_flag = hp['dim_flag']
        self.training = hp['train']
        self.iternum = 0
        self.weights_dir=hp['weights_dir']
        self.base_path = hp['base_path']
        self.save_file =hp['save_file']
        self.norm_mode = hp['norm_mode']

                
        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_max_iternum"]
        self.nt_config.stepIter = hp["nt_steps_per_loop"]
        self.nt_config.batchSize = hp["nt_batch_size"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        
        
        self.tf_config = Struct()
        self.tf_config.epochs = hp['tf_epochs']
        self.tf_config.initial_epoch = hp['tf_initial_epoch']
        self.tf_config.steps_per_epoch = hp['tf_steps_per_epoch']
        self.tf_config.batch_size = hp['tf_batch_size']
        self.tf_config.init_lr = hp['tf_init_lr']

           
        # get the other conditions (boundary)
        self.conds = self.get_conditions(*conds)
        
        # merge the inputs to generate training data
        self.train_X, self.train_Y = self.merge_inputs(data, eqns)     # 计算data loss的采样点和计算eqns loss的采样点沿行合并
        
        # Initialize the loss recording list
        self.loss_all = []
        self.val_loss = []
        self.loss_data = []
        self.loss_eqns = []
        self.loss_conds = []
        self.alpha_seq = []
        # alpha_seq is initialized by alpha
        self.alpha_seq.append(self.alpha)


        # Multi GPU or single GPU
        # 注意：为避免分布式策略冲突，统一使用单GPU模式
        # MirroredStrategy 在 Keras 3 中与自定义loss函数存在兼容性问题
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 仅使用第一个GPU，避免多GPU策略冲突
            self.tf_config.gpus_number = 1
            print(f'检测到 {len(gpus)} 个 GPU，使用单GPU模式训练（GPU:0）')
        else:
            self.tf_config.gpus_number = 1
            print('未检测到 GPU，使用 CPU 进行训练')

        self.tf_config.global_batch_size = self.tf_config.batch_size * self.tf_config.gpus_number

        # learning rate
        reduce_lr = LearningRateScheduler(self.exponential_staircase_scheduler, verbose=0)

        # callback
        uvw_log_path = os.path.join(self.weights_dir, self.savename + "_uvw_range.txt")
        # ModelCheckpoint filepath 需要指定完整的文件路径并以 .weights.h5 结尾（Keras 3 兼容）
        checkpoint_path = os.path.join(self.weights_dir, self.savename + "_checkpoint.weights.h5")
        self.call_back_list = [
            ModelCheckpoint(filepath=checkpoint_path,
                            monitor='loss', save_best_only=True, save_weights_only=True),
            reduce_lr,
            LossPrintingCallback(self, uvw_log_path, sample_num=10000)
        ]

        # optimizer - 在模型构建前创建
        self.optimizer = Adam(learning_rate=self.tf_config.init_lr)

        # 模型构建（单GPU模式，避免策略冲突）
        if self.training:
            if self.ExistModel == 0:
                self.norm_paras = hp['norm_paras']
                self.model = self.build_model()
            elif self.ExistModel == 1:
                self.norm_paras = self.loadParas()
                self.model = self.loadNN()

            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss_fn_all,
                               metrics=[self.loss_fn_data, self.loss_fn_eqns, self.loss_fn_conds])
            self.model.summary()
        # 预测模式
        else:
            if self.ExistModel == 0:
                self.norm_paras = hp['norm_paras']
                self.model = self.build_model()
            elif self.ExistModel == 1:
                self.norm_paras = self.loadParas()  
                self.model = self.loadNN()
            self.model.summary()   
      

    
    
    def build_model(self):                                           # 调用的是maps.py中定义的函数generator
        model = generator(self.layers, self.norm_paras, map_name=self.map_name, norm_mode=self.norm_mode)
        return model
  
    
    
    def get_conditions(self, *conds):                                # 从datagenerator.py中生成的边界数据中提取边界条件
        """
        get the other (boundary) conditions from the inputs
        """
        conds_keys = ['init', 'dirichlet', 
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        
        # default is no conditions
        conds_dict = dict.fromkeys(conds_keys, None)
    
        conds_num = len(conds)
        if conds_num > 0:
            # there are conditions
            for ii in np.arange(0,conds_num,1):
                # ii= 0, initial BC,             [t,x,y,z,u,v,w,p]
                # ii= 1, Dirichlet BC,           [t,x,y,z,u,v,w,p]
                # ii= 2, Neumann BC of ux,       [t,x,y,z,ux]
                # ii= 3, Neumann BC of uy,       [t,x,y,z,uy]
                # ii= 4, Neumann BC of uz,       [t,x,y,z,uz]            
                # ii= 5, Neumann BC of vx,       [t,x,y,z,vx]
                # ii= 6, Neumann BC of vy,       [t,x,y,z,vy]
                # ii= 7, Neumann BC of vz,       [t,x,y,z,vz]    
                # ii= 8, Neumann BC of wx,       [t,x,y,z,wx]
                # ii= 9, Neumann BC of wy,       [t,x,y,z,wy]
                # ii=10, Neumann BC of wz,       [t,x,y,z,wz]    
                # ii=11, Neumann BC of px,       [t,x,y,z,px]
                # ii=12, Neumann BC of py,       [t,x,y,z,py]
                # ii=13, Neumann BC of pz,       [t,x,y,z,pz]  
                if conds[ii] is None:
                    conds_dict[conds_keys[ii]] = None
                else:
                    conds_dict[conds_keys[ii]] = conds[ii]
                
        return conds_dict
            
            
    
    def merge_inputs(self, data, eqns):                               # 把计算data loss的采样点和计算eqns loss的采样点沿行方向进行合并
        """
        data points:    [t,x,y,z,u,v,w,p]
        eqns points:    [t,x,y,z]
        
        train_X:        [t,x,y,z]
        train_Y:        [t,x,y,z,u,v,w,p,flag0,flag1]
        flag0:          flag of data points, 后面flag0=1的采样点将计算data loss
        flag1:          flalg of equations points, 后面flag1=1的采样点将计算eqns loss
        
        Note: equation points include the points of boundary conditions
        """
        if data is None: 
            # there is only equation points, data points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                train_Y = np.zeros([eqns_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([eqns_num,10], dtype=np.float64) 
            train_Y[0:eqns_num,0:4] = eqns
            train_Y[0:eqns_num,9:10] = 1.0
        else:
            # there are both eqns points and data points
            data_num = data.shape[0]
            if self.dtype=='float32':
                train_Y = np.zeros([data_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([data_num,10], dtype=np.float64)             
            train_Y[0:data_num,0:8] = data
            train_Y[0:data_num,8:9] = 1.0
            # adding eqns points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                tmp = np.zeros([eqns_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                tmp = np.zeros([eqns_num,10], dtype=np.float64) 
            tmp[0:eqns_num,0:4] = eqns
            train_Y = np.concatenate([train_Y, tmp], axis=0) 
            # all the points need to estimate the residual of equations
            train_Y[:,9:10] = 1.0

        train_X = train_Y[:,0:4]
        return train_X, train_Y
    
    
  
    def ns_eqns(self, X):                                             # 被下方loss_fn_eqns函数调用，计算物理损失
        """
        Returns
        -------
        residual of Navier-Stokes equations

        显存优化说明：
        - 必须使用persistent=True，因为需要多次调用gradient()计算多个变量的梯度
        - 使用后立即del tape释放显存，避免梯度缓存长期占用
        - 这是PINN计算二阶导数的必要模式，无法用非persistent梯度带替代
        """
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)

        # 外层梯度带：用于计算二阶导数（必须persistent=True，因为要计算9个二阶导数）
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            tape1.watch(x)
            tape1.watch(y)
            tape1.watch(z)

            # 内层梯度带：用于计算一阶导数（必须persistent=True，因为要计算15个一阶导数）
            with tf.GradientTape(persistent=True) as tape2:
                # Watching gradients of t,x,y,z
                tape2.watch(t)
                tape2.watch(x)
                tape2.watch(y)
                tape2.watch(z)
                # Packing together the inputs
                X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
                # Getting the prediction
                Y = self.model(X)
                u = Y[:,0:1]
                v = Y[:,1:2]
                w = Y[:,2:3]
                p = Y[:,3:4]
                # recover to real value
                u = self.denormalize(u, 4)
                v = self.denormalize(v, 5)
                w = self.denormalize(w, 6)
                p = self.denormalize(p, 7)

            # 计算一阶导数
            u_t = tape2.gradient(u, t)
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            u_z = tape2.gradient(u, z)
            v_t = tape2.gradient(v, t)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            v_z = tape2.gradient(v, z)
            w_t = tape2.gradient(w, t)
            w_x = tape2.gradient(w, x)
            w_y = tape2.gradient(w, y)
            w_z = tape2.gradient(w, z)
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)
            p_z = tape2.gradient(p, z)

            # 关键优化：立即释放内层tape，避免显存积累
            del tape2

        # 计算二阶导数
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1.gradient(u_y, y)
        u_zz = tape1.gradient(u_z, z)
        v_xx = tape1.gradient(v_x, x)
        v_yy = tape1.gradient(v_y, y)
        v_zz = tape1.gradient(v_z, z)
        w_xx = tape1.gradient(w_x, x)
        w_yy = tape1.gradient(w_y, y)
        w_zz = tape1.gradient(w_z, z)

        # 关键优化：立即释放外层tape，避免显存积累
        del tape1

        e1 = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / self.Re) * (u_xx + u_yy + u_zz)
        e2 = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / self.Re) * (v_xx + v_yy + v_zz)
        e3 = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / self.Re) * (w_xx + w_yy + w_zz)
        e4 = u_x + v_y + w_z

        # Buidling the PINNs
        return e1, e2, e3, e4
        
        

    def get_gradient(self, X, flag):                                          # 被下方loss_fn_conds函数调用，计算边界损失
        """
        Returns
        -------
        the gradient of u,v,p

        优化说明：
        - 移除persistent=True，改为单次梯度计算
        - 每次调用只计算一个特定方向的梯度，无需persistent
        - 显著降低显存占用，特别是在边界条件计算中
        """
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)

        # 使用非persistent梯度带（单次梯度计算）
        with tf.GradientTape() as tape:
            # Watching gradients of t,x,y,z
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            # Packing together the inputs
            X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
            # Getting the prediction
            # Y = self.model(X, training=self.training)
            Y = self.model(X)
            u = Y[:,0:1]
            v = Y[:,1:2]
            w = Y[:,2:3]
            p = Y[:,3:4]
            # recover to real value
            u = self.denormalize(u, 4)
            v = self.denormalize(v, 5)
            w = self.denormalize(w, 6)
            p = self.denormalize(p, 7)

        # 计算一阶导数（根据flag选择）
        if flag.lower() == 'ux':
            g = tape.gradient(u, x)
        elif flag.lower() == 'uy':
            g = tape.gradient(u, y)
        elif flag.lower() == 'uz':
            g = tape.gradient(u, z)
        elif flag.lower() == 'vx':
            g = tape.gradient(v, x)
        elif flag.lower() == 'vy':
            g = tape.gradient(v, y)
        elif flag.lower() == 'vz':
            g = tape.gradient(v, z)
        elif flag.lower() == 'wx':
            g = tape.gradient(w, x)
        elif flag.lower() == 'wy':
            g = tape.gradient(w, y)
        elif flag.lower() == 'wz':
            g = tape.gradient(w, z)
        elif flag.lower() == 'px':
            g = tape.gradient(p, x)
        elif flag.lower() == 'py':
            g = tape.gradient(p, y)
        elif flag.lower() == 'pz':
            g = tape.gradient(p, z)

        # Buidling the PINNs
        return g
     
    
    def get_uvwp(self, X):                                                    # 被下方loss_fn_conds函数调用，计算边界损失
        """
        get the output of the network
        The predict function is designed for performance in large scale inputs. 
        For small amount of inputs that fit in one batch, directly using
        __call__() is recommended for faster execution, e.g., model(x), 
        or model(x, training=False)
        
        return the velocity and pressure

        """ 
        Xi = tf.convert_to_tensor(X[:,0:4], self.dtype)
        Y = self.model(Xi)
    
        u = self.denormalize(Y[:,0:1], 4)
        v = self.denormalize(Y[:,1:2], 5)
        w = self.denormalize(Y[:,2:3], 6)
        p = self.denormalize(Y[:,3:4], 7)
 
        return u, v, w, p
    
    
    
    def loss_fn_all(self, Y_true, Y_pred):                                    # 计算物理损失数据损失边界损失的和，总损失
        """
        自定义loss
        Y_true: [t, x, y, z, u, v, w, p, flag0, flag1]
        """
        # update the iteration number
        self.iternum = self.iternum+1 
        
        # loss of equations
        le = self.loss_fn_eqns(Y_true, Y_pred)
        # loss of data
        ld = self.loss_fn_data(Y_true, Y_pred)    
        # loss of boundary condtions
        lb = self.loss_fn_conds(Y_true, Y_pred)
        # total loss                          
        return self.alpha*le+ld+lb   
 
    
 
    def loss_fn_eqns(self, Y_true, Y_pred):                                   # 计算物理损失   这里的Ytrue和Ypred都是通过Keras model.fit() 调用的，不是显式代码给的
        """
        return the loss of eqns

        优化说明：
        - 支持分块计算PDE残差，降低显存峰值
        - 当batch_size较大时，可将X分成多个子块分别计算，最后累加损失
        - 数学上等价，但显存占用可降低3-4倍
        """
        #flag for equations
        flag_data = Y_true[:,9:10]
        num_data = tf.reduce_sum(flag_data)+1.0

        # loss of equations
        X = Y_true[:,0:4]                                              # 取了数据中txyz信息，在ns_eqns函数中调用model进行了预测，再计算梯度和物理损失

        # 分块计算PDE残差（可选优化）
        # 如果batch_size较大，可以分块计算以降低显存峰值
        chunk_size = 256  # 每块处理256个点，可根据GPU显存调整
        num_chunks = (X.shape[0] + chunk_size - 1) // chunk_size

        loss_eqns_e1_total = 0.0
        loss_eqns_e2_total = 0.0
        loss_eqns_e3_total = 0.0
        loss_eqns_e4_total = 0.0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, X.shape[0])
            X_chunk = X[start_idx:end_idx, :]
            flag_chunk = flag_data[start_idx:end_idx, :]

            # 计算该块的PDE残差
            e1, e2, e3, e4 = self.ns_eqns(X_chunk)

            # 累加该块的损失
            loss_eqns_e1_total += tf.reduce_sum(tf.square(e1 * flag_chunk))
            loss_eqns_e2_total += tf.reduce_sum(tf.square(e2 * flag_chunk))
            loss_eqns_e3_total += tf.reduce_sum(tf.square(e3 * flag_chunk))
            loss_eqns_e4_total += tf.reduce_sum(tf.square(e4 * flag_chunk))

        # 计算平均损失
        loss_eqns_e1 = loss_eqns_e1_total / num_data
        loss_eqns_e2 = loss_eqns_e2_total / num_data
        loss_eqns_e3 = loss_eqns_e3_total / num_data
        loss_eqns_e4 = loss_eqns_e4_total / num_data

        loss_eqns = loss_eqns_e1 + loss_eqns_e2 + self.z_weight * loss_eqns_e3 + loss_eqns_e4

        return loss_eqns
    
    

    def loss_fn_data(self, Y_true, Y_pred):                                   # 计算数据损失 
        """
        return the loss of data
        """       
        # loss of data
        flag_data = Y_true[:,8:9]
        num_data = tf.reduce_sum(flag_data)+1.0
        
        # the data value
        ut = Y_true[:,4:5]
        vt = Y_true[:,5:6]
        wt = Y_true[:,6:7]
        # pt = Y_true[:,7:8]
        
        # predicted value, recover to real value
    
        up = self.denormalize(Y_pred[:,0:1], 4)
        vp = self.denormalize(Y_pred[:,1:2], 5)
        wp = self.denormalize(Y_pred[:,2:3], 6)

     
        tmp = tf.square(ut-up)
        loss_data_u = tf.reduce_sum(tmp*flag_data)/num_data
        tmp = tf.square(vt-vp)
        loss_data_v = tf.reduce_sum(tmp*flag_data)/num_data
        tmp = tf.square(wt-wp)
        loss_data_w = tf.reduce_sum(tmp*flag_data)/num_data        
        
        if self.dim_flag.lower() == '2d2c':
            # 2D2C PIV, no w component
            loss_data = loss_data_u+loss_data_v  
        else:
            # consider the w component
            loss_data = loss_data_u+loss_data_v+self.z_weight*loss_data_w                # 给z方向单独加工二级权重
         
            
        return loss_data
    
    
     
    def loss_fn_conds(self, Y_true, Y_pred):                                  # 计算边界损失 
        """
        return the loss of all the BCs
        # ii= 0, initial BC,             [t,x,y,z,u,v,w,p]
        # ii= 1, Dirichlet BC,           [t,x,y,z,u,v,w,p]
        # ii= 2, Neumann BC of ux,       [t,x,y,z,ux]
        # ii= 3, Neumann BC of uy,       [t,x,y,z,uy]
        # ii= 4, Neumann BC of uz,       [t,x,y,z,uz]            
        # ii= 5, Neumann BC of vx,       [t,x,y,z,vx]
        # ii= 6, Neumann BC of vy,       [t,x,y,z,vy]
        # ii= 7, Neumann BC of vz,       [t,x,y,z,vz]    
        # ii= 8, Neumann BC of wx,       [t,x,y,z,wx]
        # ii= 9, Neumann BC of wy,       [t,x,y,z,wy]
        # ii=10, Neumann BC of wz,       [t,x,y,z,wz]    
        # ii=11, Neumann BC of px,       [t,x,y,z,px]
        # ii=12, Neumann BC of py,       [t,x,y,z,py]
        # ii=13, Neumann BC of pz,       [t,x,y,z,pz]  
        """ 
        batch_size = 1000
        loss = 0.0
        # iterate to estimate the loss of Bcs
        for key, val in self.conds.items():
            if (key == 'init') or (key == 'dirichlet'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:4]
                    # using the fluctuation to calculate the error
                    ut = val[idx,4:5]
                    vt = val[idx,5:6]
                    wt = val[idx,6:7]
                    # pt = val[idx,7:8]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    
                    tmpu = tf.reduce_mean(tf.square(ut-up))
                    tmpv = tf.reduce_mean(tf.square(vt-vp))
                    tmpw = tf.reduce_mean(tf.square(wt-wp))                   
                    loss = loss + tmpu + tmpv + tmpw
            # for the conditions of gradients: ux, uy, uz, vx, vy, vz, px, py, pz
            else:
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:4]
                    # using the fluctuation to calculate the error
                    gt = val[idx,4:5]
                    gp = self.get_gradient(tmpX, key)
                    
                    tmp = tf.reduce_mean(tf.square(gt-gp))
                    
                    loss = loss + tmp      
                    
        return loss
        
    

    
    def lbfgs_callback(self, Xi):                                            # 阶段2 LBFGS高精度迭代参数优化
        # Xi is for code running

        # iteration number
        self.iternum = self.iternum+1
        
        # prediction
        X = self.cur_train_Y[:,0:4]
        # Keras 3 移除了 workers 和 use_multiprocessing 参数
        Y_pred = self.model.predict(X, batch_size=8192)                        # 阶段2 基于当前权重的前向预测
        
        # loss of equations
        le = self.loss_fn_eqns(self.cur_train_Y, Y_pred)
        # loss of the data
        ld = self.loss_fn_data(self.cur_train_Y, Y_pred)    
        # loss of other conditions (BCs)
        lb = self.loss_fn_conds(self.cur_train_Y, Y_pred)
        # total loss                          
        loss =  self.alpha*le+ld+lb
    
        self.loss_data.append(ld)
        self.loss_eqns.append(le)  
        self.loss_conds.append(lb)
        self.loss.append(loss)
        if self.iternum % 10 == 0:        
            print('L-BGFS-B Iter=%05d: Loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e' %
                      (self.iternum, loss, ld, le, lb))                
      
        

    def train(self):                                                         # 训练过程，先用 TensorFlow 的 Adam 训练 PINN，
        self.training = True                                                 # 让模型收敛到合理范围，再用 LBFGS 做高精度收敛，最后保存模型与训练过程
        # 注意：Keras 3 移除了 workers 和 use_multiprocessing 参数
        history = self.model.fit(self.train_X, self.train_Y, batch_size=self.tf_config.global_batch_size,
                                 epochs=self.tf_config.epochs,
                                 verbose=0,
                                 callbacks=self.call_back_list,
                                 validation_split=0.0,
                                 shuffle=True,
                                 initial_epoch=self.tf_config.initial_epoch,
                                 steps_per_epoch=self.tf_config.steps_per_epoch)
        
        self.loss_data = history.history['loss_fn_data']
        self.loss_eqns = history.history['loss_fn_eqns']
        self.loss_conds = history.history['loss_fn_conds']
        self.loss = history.history['loss']
        # self.val_loss = history.history['val_loss']
        
        # save model
        self.saveNN()  
        # save the parameters
        sio.savemat(self.save_file +'_paras.mat', 
                    {'batch_size':self.tf_config.batch_size,
                     'epochs':self.tf_config.epochs,
                     'loss':self.loss,
                     'val_loss':self.val_loss,
                     'loss_data':self.loss_data,     
                     'loss_eqns':self.loss_eqns,
                     'loss_conds':self.loss_conds,
                     'alpha':self.alpha,
                     'alpha_seq':self.alpha_seq,
                     'norm_paras':self.norm_paras})
        
        if self.nt_config.maxIter > 0:
            # reload the model to a single GPU
            self.model = self.loadNN()
            # L-BFGS training
            loopnum = np.fix(self.nt_config.maxIter/self.nt_config.stepIter)
            # starting
            for ii in np.arange(0,loopnum,1):
                # randomly select the training data
                if self.nt_config.batchSize > 0:
                    idx = np.random.choice(self.train_X.shape[0], self.nt_config.batchSize, replace=True)
                    self.cur_train_X = self.train_X[idx,:]
                    self.cur_train_Y = self.train_Y[idx,:]
                    
                elif self.nt_config.batchSize == 0:
                    self.cur_train_X = self.train_X
                    self.cur_train_Y = self.train_Y
                        
                # Transforms model into a function of its parameter
                func, params, names = tf_function_factory(self.model, self.loss_fn_all, self.cur_train_X, self.cur_train_Y)
                # Minimization
                res = minimize(func, 
                                params, 
                                method='L-BFGS-B',
                                options={'disp':None,
                                        'maxiter': self.nt_config.stepIter,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'gtol':1e-8,
                                        'eps':1e-8,
                                        'ftol': self.nt_config.tolFun},
                                callback= self.lbfgs_callback)  
        
            # save the model
            self.saveNN()
            # save the parameters
            sio.savemat(self.save_file+'_paras.mat', 
                        {'batch_size':self.tf_config.batch_size,
                         'epochs':self.tf_config.epochs,
                         'loss':self.loss,
                         'loss_data':self.loss_data,     
                         'loss_eqns':self.loss_eqns,
                         'loss_conds':self.loss_conds,
                         'alpha':self.alpha,
                         'alpha_seq':self.alpha_seq,
                         'norm_paras':self.norm_paras})
        
        return history
    
        
    def predict(self, input_X):
        # prediction
        # Keras 3 移除了 workers 和 use_multiprocessing 参数
        Y = self.model.predict(input_X, batch_size=8196*self.tf_config.gpus_number)
        u = Y[:,0:1]
        v = Y[:,1:2]
        w = Y[:,2:3]
        p = Y[:,3:4]        
        # recover to real value

        u = self.denormalize(Y[:,0:1], 4)
        v = self.denormalize(Y[:,1:2], 5)
        w = self.denormalize(Y[:,2:3], 6)
        p = self.denormalize(Y[:,3:4], 7)

        return u, v, w, p
    
      
    def saveNN(self):                                                         # 保存模型，被train()调用
        """
        仅保存模型权重，不保存完整模型（避免保存优化器状态）

        优化说明：
        - 改为save_weights()而非save()，避免保存Adam优化器的一阶/二阶动量参数
        - 断点续训时，优化器会被重新初始化，避免历史动量参数叠加导致的显存溢出
        - 权重文件更小，加载更快
        - Keras 3 需要使用 .weights.h5 格式
        """
        # 确保权重目录存在
        os.makedirs(self.weights_dir, exist_ok=True)

        # 保存权重文件（Keras 3 使用 .weights.h5 格式）
        weight_path = os.path.join(self.weights_dir, self.savename + ".weights.h5")
        self.model.save_weights(weight_path)
        print(f"✔ 模型权重已保存至: {weight_path}")

    def loadNN(self):                                                         # 加载模型，被train()调用
        """
        加载模型权重并重新编译（不加载优化器状态）

        优化说明：
        - 先构建新模型，再加载权重，避免加载历史优化器状态
        - 重新初始化优化器，确保Adam的一阶/二阶动量参数为零
        - 这是解决断点续训显存异常的关键步骤
        - Keras 3 需要使用 .weights.h5 格式
        """
        # 权重文件路径（Keras 3 使用 .weights.h5 格式）
        weight_path = os.path.join(self.weights_dir, self.savename + ".weights.h5")

        # 检查权重文件是否存在（兼容旧格式）
        old_weight_path = os.path.join(self.weights_dir, self.savename)
        if not os.path.exists(weight_path):
            # 尝试旧格式 (TensorFlow checkpoint)
            if os.path.exists(old_weight_path + ".index"):
                weight_path = old_weight_path
                print(f"⚠ 使用旧格式权重文件: {weight_path}")
            else:
                raise FileNotFoundError(f"权重文件缺失: {weight_path}")

        # 第一步：构建新模型（不加载任何历史状态）
        model = self.build_model()

        # 第二步：加载权重
        model.load_weights(weight_path)
        print(f"✔ 已从 {weight_path} 加载模型权重")

        # 第三步：重新初始化优化器（关键！避免加载历史动量参数）
        self.optimizer = Adam(learning_rate=self.tf_config.init_lr)
        print(f"✔ 优化器已重新初始化（清除历史动量参数）")

        return model

    def loadParas(self):                                                      # 加载模型参数
        """使用统一管理的路径加载参数"""
        file_path = os.path.join(self.weights_dir, f"{self.savename}_paras.mat")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"参数文件不存在: {file_path}")
            
        data = sio.loadmat(file_path)
        return data['norm_paras']
        
    
  
    def denormalize(self, Y, index):                                          # 逆归一化函数
        if self.norm_mode in ['none', 'input_only']:
            return Y
        elif self.norm_mode == 'minmax':
            vmin = self.norm_paras[0,index]
            vmax = self.norm_paras[1,index]
            return Y*(vmax - vmin) + vmin
        elif self.norm_mode == 'standard':
            mean = self.norm_paras[0,index]
            std  = self.norm_paras[1,index]
            return Y*std + mean

    
    def exponential_continuous_scheduler(self, epoch):                         # 多种学习率下降策略
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate,(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr 
                          
    def exponential_staircase_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate, np.floor(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr      
            
    def constant_scheduler(self, epoch):
        """
        constant learning rate decay
        """
        return self.tf_config.init_lr   
    
    def piecewise_scheduler(self, epoch):
        """
        piecewise learning rate decay
        """
        rate = np.floor(epoch/100)
        return self.tf_config.init_lr/(rate+1.0)

class Config():
    def __init__(self):

        ## Transformer Configs ##
        self.numtasks = 31 # MT30 has 30 tasks, + 1 for unknown task we will learn
        self.latent_dim = 512
        self.action_dim = 61
        self.use_horizon_batchsize_dimensioning = False
        self.only_state_p = 0.5
        self.multitask = True
        self.obs = 'state'
        self.obs_shape = {'state': [151]}
        self.tasks=[]
        for i in range(30):
            self.tasks.append(i)
        self.tasks.append('humanoid_h1hand-sit_simple-v0')
        self.task_dim= 64 # 96 if mt80 else 64. 0 if single task,
        self.num_enc_layers = 2
        self.enc_dim = 2
        self.simnorm_dim = 8
        self.encoder_dim = 256

        ## Training args ##
        self.lr = 1e-4
        self.num_epochs = 120
        self.batch_size = 64
        self.clip_grad_norm = 20

        ## Used in TMPC2 Loss ##
        self.rho = .5
        self.entropy_coef = 1e-4
        self.tau = 0.005 # Used in the running scale for Qs
        self.horizon = 4

        self.consistency_coef = 20
        self.reward_coef = 0.1
        self.value_coef = 0.1
        self.action_coef = 0.1
        self.log_pi_coef = .1

        ## Used in tdmpc2 math file ##
        self.num_bins = 101
        self.vmin = -10
        self.vmax = 10
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
        
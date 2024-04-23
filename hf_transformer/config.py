class Config():
    def __init__(self):
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
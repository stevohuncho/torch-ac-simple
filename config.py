class Config:
    def __init__(self, 
        epochs = 4, 
        batch_size=256, 
        frames_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        optim_alpha=0.99,
        optim_eps=1e-08,
        clip_eps=0.2,
        recurrence=1,
        use_memory=False,
        use_text=False,
        reshape_reward=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.frames_per_proc = frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optim_alpha = optim_alpha
        self.optim_eps = optim_eps
        self.clip_eps = clip_eps
        self.recurrence = recurrence
        self.use_memory = use_memory
        self.use_text = use_text
        self.reshape_reward = reshape_reward


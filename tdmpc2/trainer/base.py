class Trainer:
    """Base trainer class for TD-MPC2."""

    def __init__(self, cfg, env, agent, buffer, logger, cond_sampler):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.cond_sampler = cond_sampler

        print("Architecture:", self.agent.model)
        print("Learnable parameters: {:,}".format(self.agent.model.total_params))
        print("Condition sampler:", self.cond_sampler)

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        raise NotImplementedError

    def train(self):
        """Train a TD-MPC2 agent."""
        raise NotImplementedError

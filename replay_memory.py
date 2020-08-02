import numpy as np

class ReplayMemory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.batch_gae_return = []  # Gets populate when GAE is calculated
        self.batch_v = []  # Gets populated when agent decides action
        self.batch_log_probs = []  # Gets populated when agent decides action

    def get_batch(self, batch_size):
        """Returns a randomized batch of experiences relevant to training
        """
        # generates random mini-batches until we have covered the full batch
        buffer_size = len(self.batch_s)
        # for _ in range(buffer_size // batch_size):
        #     rand_ids = np.random.choice(buffer_size, size=batch_size, replace=False)

        #     yield np.asarray(self.batch_s)[rand_ids, :], np.asarray(self.batch_a)[rand_ids, :], np.asarray(self.batch_log_probs)[rand_ids, :], np.asarray(self.batch_gae_return)[rand_ids, :], np.asarray(self.batch_v)[rand_ids,:]
        
        idxs = np.arange(buffer_size)
        np.random.shuffle(idxs)
        for start in range(0, buffer_size, batch_size):
            end = start + batch_size
            yield np.asarray(self.batch_s)[start:end, :], np.asarray(self.batch_a)[start:end, :], np.asarray(self.batch_log_probs)[start:end, :], np.asarray(self.batch_gae_return)[start:end, :], np.asarray(self.batch_v)[start:end,:]

        # states = np.asarray(self.batch_s)
        # actions = np.asarray(self.batch_a)
        # log_probs = np.asarray(self.batch_log_probs)
        # gae_returns = np.asarray(self.batch_gae_return)
        # values = np.asarray(self.batch_v)
        
        # return states, actions, log_probs, gae_returns, values, idxs

    def store(self, s, a, r, s_, done):
        """Stores transitons observed by the agent
        """        
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.batch_gae_return.clear()
        self.batch_v.clear()

    def __len__(self):
        return len(self.batch_s)
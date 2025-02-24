import numpy as np
import torch
import algs.core_discrete as core

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for discrete NAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size), dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.trm_buf = np.zeros(size, dtype=np.float32)
        self.trc_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, trm, trc):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.trm_buf[self.ptr] = trm
        self.trc_buf[self.ptr] = trc
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     trm=self.trm_buf[idxs],
                     trc=self.trc_buf[idxs])
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        return {k: torch.as_tensor(v, dtype=torch.float32) if k != 'act' else torch.as_tensor(v, dtype=torch.int64) for k, v in batch.items()}
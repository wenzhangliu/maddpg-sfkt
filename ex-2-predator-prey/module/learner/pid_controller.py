import numpy as np
import time


class controller():
    def __init__(self, obs_shape, dim_act, n_agents):
        self.n_agents = n_agents
        self.K_pid = [3.0, 0.1, 0.1]
        self.Kp = [self.K_pid[0], self.K_pid[0]]
        self.Ki = [self.K_pid[1], self.K_pid[1]]
        self.Kd = [self.K_pid[2], self.K_pid[2]]
        self.input_ref = np.zeros([2], dtype=np.float)
        self.error = np.zeros([3, 2])   # e_{t-2}, e_{t-1}, e_{t}, dim_pos = 2
        self.action = np.zeros(dim_act)

    def pid_cal(self, pos_error):
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]
        self.error[0] = pos_error
        self.action_delta = np.multiply(self.Kp, (self.error[0] - self.error[1])) + \
                            np.multiply(self.Ki, self.error[0]) + \
                            np.multiply(self.Kd, (self.error[0] - 2 * self.error[1] + self.error[2]))

        action_r = self.action[0] + self.action_delta[0]
        action_u = self.action[1] + self.action_delta[1]

        self.action[0] = action_r
        self.action[1] = action_u

        return self.action

    def get_actions(self, obs, sess=None, noise=False):
        index_1 = - ((self.n_agents-1) * 2 + 2)
        index_2 = - (self.n_agents-1) * 2
        pos_error = obs[0][index_1:index_2]
        action = self.pid_cal(pos_error)
        return action

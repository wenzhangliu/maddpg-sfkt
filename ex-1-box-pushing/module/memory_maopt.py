import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape, reward_shape, option_shape,
                 state_shape=None,
                 Q_value_shape=None, act_adv_shape=None,
                 Q_old_shape=None, act_old_shape=None):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=reward_shape)
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)
        self.options = RingBuffer(limit, shape=option_shape)
        self.actions_advice = RingBuffer(limit, shape=action_shape)

        # for mixing network
        if (state_shape == None):
            self.is_mixing = False
        else:
            self.is_mixing = True
            self.states0 = RingBuffer(limit, shape=state_shape)
            self.states1 = RingBuffer(limit, shape=state_shape)

        # for transfer learning
        if (Q_value_shape==None) or (act_adv_shape==None):
            self.is_transfer = False
        else:
            self.is_transfer = True

            self.Q_value_advice = RingBuffer(limit, shape=Q_value_shape)
            self.actions_advice = RingBuffer(limit, shape=act_adv_shape)

            self.Q_value_old = RingBuffer(limit, shape=Q_old_shape)
            self.actions_old = RingBuffer(limit, shape=act_old_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        option_batch = self.options.get_batch(batch_idxs)
        a_advice_batch = self.actions_advice.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'options': array_min2d(option_batch),
            'a_advice': array_min2d(a_advice_batch),
            'terminals1': array_min2d(terminal1_batch)
        }

        # for transfer learning
        if self.is_transfer:
            q_advice_batch = self.Q_value_advice.get_batch(batch_idxs)
            a_advice_batch = self.actions_advice.get_batch(batch_idxs)

            q_old_batch = self.Q_value_old.get_batch(batch_idxs)
            a_old_batch = self.actions_old.get_batch(batch_idxs)

            result_advice = {'q_advice': array_min2d(q_advice_batch),
                'a_advice': array_min2d(a_advice_batch),

                'q_old': array_min2d(q_old_batch),
                'a_old': array_min2d(a_old_batch)}
            result.update(result_advice)

        if self.is_mixing:
            state0_batch = self.states0.get_batch(batch_idxs)
            state1_batch = self.states1.get_batch(batch_idxs)
            result_state = {'state0': array_min2d(state0_batch),
                            'state1': array_min2d(state1_batch)}
            result.update(result_state)

        return result

    def append(self, obs0, action, reward, obs1, terminal1, options,
               state0=None, state1=None,
               q_adv=None, a_adv=None, q_old=None, a_old=None,
               training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.options.append(options)
        self.actions_advice.append(a_adv)
        self.terminals1.append(terminal1)

        # for transfer learning
        if self.is_transfer:
            self.Q_value_advice.append(q_adv)
            self.actions_advice.append(a_adv)

            self.Q_value_old.append(q_old)
            self.actions_old.append(a_old)
        if self.is_mixing:
            self.states0.append(state0)
            self.states1.append(state1)

    @property
    def nb_entries(self):
        return len(self.observations0)

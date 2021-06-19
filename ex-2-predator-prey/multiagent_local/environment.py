import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent_local.multi_discrete import MultiDiscrete
from copy import deepcopy


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # window size
        self.window_width = 500  # 500
        self.window_height = 500  # 500

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # In adversarial case, reward_n don't share
        done_n = np.reshape(done_n, [done_n.__len__(), 1])

        ## update the states after collisions
        self.agent_box_collision()
        # self.box_landmark_collision()

        # update agents' and box's state after collision
        for agent in self.agents:
            self.update_agent_state(agent)
        for box in self.world.boxes:
            self.update_box(box)
        return obs_n, reward_n, done_n, info_n

    def get_global_state(self):
        state = []
        for idx_entity, entity in enumerate(self.world.entities):
            state.append(entity.state.p_pos)
            state.append(entity.state.p_vel)

        global_state = np.reshape(state, [-1])
        return global_state

    def get_global_state_size(self):
        return self.world.entities.__len__() * 2 * 2

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def reset_agent(self):
        # reset world
        self.reset_callback(self.world, reset_landmark=False)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def move_landmark(self, which_landmark, center, radius, angle):
        # get landmarks' position
        self.world.landmarks[which_landmark].state.p_pos = center + [radius * np.cos(angle), radius * np.sin(angle)]

    def move_landmark_rand(self, which_landmark):
        self.world.landmarks[which_landmark].state.p_pos = 2.0 * np.random.rand(2) - 1.0

    def move_landmark_exchange(self, which_landmark):
        # get landmarks' position
        # change pos
        pos = self.world.landmarks[which_landmark].state.p_pos
        self.world.landmarks[which_landmark].state.p_pos = self.world.landmarks[which_landmark + 1].state.p_pos
        self.world.landmarks[which_landmark + 1].state.p_pos = pos

        # self.world.landmarks[which_landmark].state.p_pos= 2.0 * np.random.rand(2) - 1.0

    def move_landmark_hor(self, which_landmark, center, radius, angle):
        # get landmarks' position
        self.world.landmarks[which_landmark].state.p_pos = center + [radius * np.cos(angle), 0]

    def move_landmark_vec(self, which_landmark, center, radius, angle):
        # get landmarks' position
        self.world.landmarks[which_landmark].state.p_pos = center + [0, radius * np.sin(angle)]

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        # return self.done_callback(agent, self.world)
        return self.done_callback(agent)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', angle=0):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent_local import rendering
                self.viewers[i] = rendering.Viewer(self.window_width, self.window_height)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent_local import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if 'agent' in entity.name:
                    geom = rendering.make_circle(entity.size)
                    xform = rendering.Transform()
                    geom.set_color(*entity.color, alpha=0.5)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                    #
                    # geom = rendering.make_arrow(len=entity.size*2)
                    # xform = rendering.Transform()
                    # geom.set_color(*entity.color, alpha=0.5)
                    # geom.add_attr(xform)
                    # self.render_geoms.append(geom)
                    # self.render_geoms_xform.append(xform)
                elif 'box' in entity.name:
                    geom = rendering.make_circle(entity.size)
                    xform = rendering.Transform()
                    geom.set_color(*entity.color, alpha=0.95)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                elif 'landmark' in entity.name:
                    geom = rendering.make_circle(entity.size)
                    xform = rendering.Transform()
                    geom.set_color(*entity.color, alpha=0.95)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                else:
                    geom = rendering.make_triangle(entity.size)
                    xform = rendering.Transform()
                    geom.set_color(*entity.color)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                # geom.add_attr(xform)
                # self.render_geoms.append(geom)
                # self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent_local import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.world.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

        if agent.wall:
            if (agent.state.p_pos[0] + agent.size) > self.world.move_range_agent:
                agent.state.p_pos[0] = 2 * (self.world.move_range_agent-0.8*agent.size) - agent.state.p_pos[0]
                agent.state.p_vel[0] = - agent.state.p_vel[0]
                # agent.action.u[0] = - agent.action.u[0]
            if (agent.state.p_pos[0] - agent.size) < -self.world.move_range_agent:
                agent.state.p_pos[0] = -2 * (self.world.move_range_agent-0.8*agent.size) - agent.state.p_pos[0]
                agent.state.p_vel[0] = - agent.state.p_vel[0]
                # agent.action.u[0] = - agent.action.u[0]
            if (agent.state.p_pos[1] + agent.size) > self.world.move_range_agent:
                agent.state.p_pos[1] = 2 * (self.world.move_range_agent-0.8*agent.size) - agent.state.p_pos[1]
                agent.state.p_vel[1] = - agent.state.p_vel[1]
                # agent.action.u[1] = - agent.action.u[1]
            if (agent.state.p_pos[1] - agent.size) < -self.world.move_range_agent:
                agent.state.p_pos[1] = -2 * (self.world.move_range_agent-0.8*agent.size) - agent.state.p_pos[1]
                agent.state.p_vel[1] = - agent.state.p_vel[1]
                # agent.action.u[1] = - agent.action.u[1]

    def update_box(self, box):
        # set communication state (directly for now)
        if box.silent:
            box.state.c = np.zeros(self.world.dim_c)
        else:
            noise = np.random.randn(*box.action.c.shape) * box.c_noise if box.c_noise else 0.0
            box.state.c = box.action.c + noise

        if box.wall:
            wall_range = 1.0
            if (box.state.p_pos[0] + box.size) > wall_range:
                box.state.p_pos[0] = 2 * (wall_range-box.size) - box.state.p_pos[0]
                box.state.p_vel[0] = - box.state.p_vel[0]
                # box.action.u[0] = box.action.u[0] * 0.1
            if (box.state.p_pos[0] - box.size) < -wall_range:
                box.state.p_pos[0] = -2 * (wall_range-box.size) - box.state.p_pos[0]
                box.state.p_vel[0] = - box.state.p_vel[0]
                # box.action.u[0] = box.action.u[0] * 0.1
            if (box.state.p_pos[1] + box.size) > wall_range:
                box.state.p_pos[1] = 2 * (wall_range-box.size) - box.state.p_pos[1]
                box.state.p_vel[1] = - box.state.p_vel[1]
                # box.action.u[1] = box.action.u[1] * 0.1
            if (box.state.p_pos[1] - box.size) < -wall_range:
                box.state.p_pos[1] = -2 * (wall_range-box.size) - box.state.p_pos[1]
                box.state.p_vel[1] = - box.state.p_vel[1]
                # box.action.u[1] = box.action.u[1] * 0.1

    def agent_box_collision(self):
        pos_next = np.zeros([self.world.agents.__len__(), 2])
        vel_next = np.zeros([self.world.agents.__len__(), 2])
        force_box = np.zeros([self.world.agents.__len__(), 2])
        for a, agent in enumerate(self.world.agents):
            for b, box in enumerate(self.world.boxes):
                delta_pos = agent.state.p_pos - box.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + box.size
                if dist > dist_min:
                    box.collide_with_agents[int(agent.name[-1])] = False
                    continue
                box.collide_with_agents[int(agent.name[-1])] = True
                ## calculate the positions and velosity after collision for agents
                d_square = np.square(delta_pos)
                dxdy = delta_pos[0] * delta_pos[1]
                d_square_sum = d_square[0] + d_square[1]
                d_square_dif = d_square[0] - d_square[1]

                ## get collide forces for box
                k = self.world.contact_margin
                penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
                force_box[a] = -self.world.contact_force * delta_pos / dist * penetration
                ## get next velosities for agent
                if 0.0 == d_square_sum:
                    continue
                vel_next[a, 0] = (-d_square_dif * agent.state.p_vel[0] - 2 * dxdy * agent.state.p_vel[1]) / d_square_sum
                vel_next[a, 1] = (d_square_dif * agent.state.p_vel[1] - 2 * dxdy * agent.state.p_vel[0]) / d_square_sum
                agent.state.p_vel = deepcopy(vel_next[a])
                ## get next positions for agent
                # if box.size > dist + agent.size: beyond_dist = 2 * (box.size - dist)
                # else: beyond_dist = 2 * (agent.size + box.size - dist)
                beyond_dist = box.size + agent.size - dist
                sin_theta = delta_pos[1] / np.sqrt(d_square_sum)
                cos_theta = delta_pos[0] / np.sqrt(d_square_sum)
                pos_next[a, 0] = agent.state.p_pos[0] + beyond_dist * cos_theta
                pos_next[a, 1] = agent.state.p_pos[1] + beyond_dist * sin_theta
                agent.state.p_pos = deepcopy(pos_next[a])

        ## update box states
        for b, box in enumerate(self.world.boxes):
            if all(box.collide_with_agents) is True:
                box.state.p_vel = box.state.p_vel * (1 - self.world.damping)
                force = np.sum(force_box, axis=0)
                box.state.p_vel += (force / box.real_mass) * self.world.dt
                box.state.p_pos += box.state.p_vel * self.world.dt

        return

    def box_landmark_collision(self):
        pos_next = np.zeros([self.world.boxes.__len__(), 2])
        vel_next = np.zeros([self.world.boxes.__len__(), 2])
        force_landmark = np.zeros([self.world.boxes.__len__(), 2])
        for b, box in enumerate(self.world.boxes):
            for l, landmark in enumerate(self.world.landmarks):
                delta_pos = box.state.p_pos - landmark.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = box.size + landmark.size
                if dist > dist_min:
                    landmark.collide_with_boxes[int(box.name[-1])] = False
                    continue
                landmark.collide_with_boxes[int(box.name[-1])] = True
                ## calculate the positions and velosity after collision for boxes
                d_square = np.square(delta_pos)
                dxdy = delta_pos[0] * delta_pos[1]
                d_square_sum = d_square[0] + d_square[1]
                d_square_dif = d_square[0] - d_square[1]

                ## get collide forces for landmark
                k = self.world.contact_margin
                penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
                force_landmark[b] = -self.world.contact_force * delta_pos / dist * penetration
                ## get next velosities for box
                if 0.0 == d_square_sum:
                    continue
                vel_next[b, 0] = (-d_square_dif * box.state.p_vel[0] - 2 * dxdy * box.state.p_vel[1]) / d_square_sum
                vel_next[b, 1] = (d_square_dif * box.state.p_vel[1] - 2 * dxdy * box.state.p_vel[0]) / d_square_sum
                box.state.p_vel = deepcopy(vel_next[b])
                ## get next positions for box
                # if landmark.size > dist + box.size: beyond_dist = 2 * (landmark.size - dist)
                # else: beyond_dist = 2 * (box.size + landmark.size - dist)
                beyond_dist = landmark.size + box.size - dist
                sin_theta = delta_pos[1] / np.sqrt(d_square_sum)
                cos_theta = delta_pos[0] / np.sqrt(d_square_sum)
                pos_next[b, 0] = box.state.p_pos[0] + beyond_dist * cos_theta
                pos_next[b, 1] = box.state.p_pos[1] + beyond_dist * sin_theta
                box.state.p_pos = deepcopy(pos_next[b])

        # ## update landmark states
        # for l, landmark in enumerate(self.world.landmarks):
        #     if all(landmark.collide_with_boxes) is True:
        #         landmark.state.p_vel = landmark.state.p_vel * (1 - self.world.damping)
        #         force = np.sum(force_landmark, axis=0)
        #         landmark.state.p_vel += (force / landmark.real_mass) * self.world.dt
        #         landmark.state.p_pos += landmark.state.p_vel * self.world.dt

        return


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

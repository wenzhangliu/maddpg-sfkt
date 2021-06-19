import numpy as np
from multiagent_local.core import World, Agent, Landmark, Box, Obstacle
from multiagent_local.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_agt=2, target_idx=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agt
        num_boxes = 1
        num_adversaries = 0
        self.target_idx = target_idx
        self.init_pos_landmark = np.array([[0, -0.15],
                                           [0, 0.15],
                                           [0.40, 0]
                                           ])
        self.num_landmarks = len(self.init_pos_landmark)
        self.size_agent = 0.04
        density_agent = 500
        self.size_landmark = 0.06
        density_landmark = 40
        self.size_box = 0.20
        density_box = 500
        self.move_range = 8.0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.size = self.size_agent
            agent.color = np.array([0.20 * (i + 1), 0.20 * (i + 3.5), 0.20 * (i + 1)])
            agent.density = density_agent
            agent.real_mass = agent.mass
            agent.collide = True
            agent.silent = True
            agent.wall = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        # add boxes
        world.boxes = [Box() for i in range(num_boxes)]
        for i, box in enumerate(world.boxes):
            box.name = 'box %d' % i
            box.size = self.size_box
            box.density = density_box
            box.real_mass = box.mass
            box.color = np.array([0.7, 0.8, 0.8])
            box.index = i
            box.collide = True
            box.silent = True
            box.wall = True
            box.collide_with_agents = [False for i in range(num_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.size = self.size_landmark
            landmark.density = density_landmark
            landmark.real_mass = landmark.mass
            landmark.color = np.array([0.8, 0.4, 0.4])
            if self.target_idx == i: landmark.color = np.array([0.8, 0.1, 0.1])
            landmark.index = i
            landmark.collide = False
            landmark.movable = False
            landmark.wall = True
            landmark.collide_with_boxes = [False for i in range(num_boxes)]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1 + self.size_box, 1 - self.size_box, world.dim_p)
            landmark.state.p_pos = self.init_pos_landmark[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, box in enumerate(world.boxes):
            # box.state.p_pos = np.random.uniform(-1+box.size+self.size_agent, 1-box.size-self.size_agent, world.dim_p)
            pos_y = np.random.uniform(-1 + box.size + self.size_agent * 6, 1 - box.size - self.size_agent * 6, 1)
            # pos_y = 0.0
            box.state.p_pos = np.array([-1 + box.size + self.size_agent * 8, pos_y])
            box.state.p_vel = np.zeros(world.dim_p)
        for agent in world.agents:
            wrong_pos = True
            while wrong_pos:
                agent.state.p_pos = np.random.uniform(-1 + agent.size, +1 - agent.size, world.dim_p)
                d_agt_box = np.linalg.norm(agent.state.p_pos - world.boxes[0].state.p_pos)
                if d_agt_box >= (agent.size + world.boxes[0].size):
                    wrong_pos = False
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        rew = np.zeros([self.num_landmarks + 1], dtype=np.float)
        # collide with other agents
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and (a.name != agent.name):
                    rew[-1] -= 1.0  # penalty for colliding with each other.
            if self.collide_wall(agent):
                rew[-1] -= 1.0
        for box in world.boxes:
            if self.collide_wall(box):
                rew[-1] -= 1.0
            if self.is_collision(agent, box):
                rew[-1] += 1.0  # reward for catching box.

        for i, box in enumerate(world.boxes):
            for l, landmark in enumerate(world.landmarks):
                # if l == self.target_idx:
                # continuous reward
                # rew[l] = (- 5.0 * np.linalg.norm(box.state.p_pos - landmark.state.p_pos))
                # sparse reward
                if self.is_collision(box, landmark):
                    rew[l] += 5  # 10.0
                # if 1 == l:
                #     rew[l] = (-5.0 * np.linalg.norm(box.state.p_pos - landmark.state.p_pos))

        # rew[self.target_idx] = rew[self.target_idx] + rew[-1]

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_pos = []
        box_landmark_pos = []
        for entity in world.landmarks:  # world.entities:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)
            box_landmark_pos.append(world.boxes[0].state.p_pos - entity.state.p_pos)
        # get positions and velocity of all entities in this agent's reference frame
        box_pos, box_vel = [], []
        for entity in world.boxes:
            box_pos.append(entity.state.p_pos - agent.state.p_pos)
            box_vel.append(entity.state.p_vel - agent.state.p_vel)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            # obs = np.concatenate([agent.state.p_vel] + landmark_pos + box_pos + other_pos)
            # obs = np.append(obs, dis_land_box)
            # return obs
            return np.concatenate(
                [agent.state.p_vel] + box_pos + box_vel + box_landmark_pos + landmark_pos + other_pos)
            # return np.concatenate(
            #     [agent.state.p_vel] + landmark_pos + box_pos + box_vel + other_pos)
        else:
            # other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + landmark_pos + box_pos + box_vel + other_pos)

    def collide_wall(self, agent):
        position = agent.state.p_pos
        if (abs(position[0]) > 1.0) or (abs(position[1]) > 1.0):
            return True
        else:
            return False

    def done(self, agent):
        dist = np.linalg.norm(agent.state.p_pos)
        if dist > self.move_range:
            return True
        else:
            return False

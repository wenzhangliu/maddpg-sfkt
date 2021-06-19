import numpy as np
from multiagent_local.core import World, Agent, Landmark
from multiagent_local.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_agt=2, target_idx=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_prey = 1  # faster
        num_predator = num_agt - num_prey  # slower
        num_agents = num_agt
        num_landmarks = 0
        size_predator = 0.08
        size_prey = 0.2
        size_landmark = 0.25
        # self.capture_penalty = -5.0
        self.capture_reward = +10.0
        self.num_tasks = num_predator
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.wall = True
            agent.is_predator = True if i < num_predator else False
            agent.color = np.array([0.85, 0.35, 0.35]) if not agent.is_predator else np.array(
                [0.20 * (i + 1), 0.20 * (i + 3.5), 0.20 * (i + 1)])
            agent.size = size_predator if agent.is_predator else size_prey
            agent.accel = 3.0 if agent.is_predator else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.is_predator else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.collide = True
            landmark.movable = False
            landmark.size = size_landmark
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.is_predator:
            collisions = 0
            for a in self.preys(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent1.size + agent2.size
        return True if (dist < dist_min) else False

    # return all agents that are not adversaries
    def predators(self, world):
        return [agent for agent in world.agents if not agent.is_predator]

    # return all adversarial agents
    def preys(self, world):
        return [agent for agent in world.agents if agent.is_predator]

    def reward(self, agent, world):
        return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        rew = np.zeros([self.num_tasks], dtype=np.float32)

        predators = self.predators(world)
        preys = self.preys(world)
        count_capture = 0

        for predator in predators:
            for prey in preys:
                if self.is_collision(predator, prey):
                    count_capture += 1
        if count_capture == 1:
            rew[0] += self.capture_reward
        if count_capture == 2:
            rew[1] += self.capture_reward
        if count_capture >= 3:
            rew[2] += self.capture_reward

        return rew

    def prey_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        predators = self.predators(world)
        if agent.collide:
            for a in predators:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def predator_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        preys = self.preys(world)
        predators = self.predators(world)
        if agent.collide:
            for ag in preys:
                for adv in predators:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
            # if not other.adversary:
            #     other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def done(self, agent):
        dist = np.linalg.norm(agent.state.p_pos)
        if dist > 2.0:
            return True
        else:
            return False

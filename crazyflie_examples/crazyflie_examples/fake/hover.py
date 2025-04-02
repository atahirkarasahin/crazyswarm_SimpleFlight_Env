import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
#from torchrl.data import Unbounded, Composite, Categorical, Bounded

from tensordict.tensordict import TensorDict, TensorDictBase

class FakeHover(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 1
        super().__init__(cfg, connection, swarm)
        self.policy_select = 1 # 0: simpleflight, 1: own policy

        self.target_pos = torch.tensor([[0., 0., 1.25]])
        self.max_episode_length = 500

    def _set_specs(self):
        
        if self.policy_select == 0:
            observation_dim = 19 # position, quat, linear velocity, heading, lateral, up
            self.time_encoding = True
        else:
            observation_dim = 15 # position, quat, linear velocity, heading, lateral, up
            self.time_encoding = False

        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
            state_dim = observation_dim
            print("obs dim", observation_dim)
        else:
            state_dim = observation_dim + 4

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "state": UnboundedContinuousTensorSpec((1, state_dim),device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action":  BoundedTensorSpec(-1, 1, 4, device=self.device).unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.rpos = self.target_pos - self.drone_state[..., :3]
        #print("rpos:", self.rpos)

        # hover_rapid
        if self.policy_select == 0:
            if self.time_encoding:
                obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 19:28], torch.zeros((self.num_cf, 4))]

                # obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 19:28]]
                # t = (self.progress_buf / self.max_episode_length) * torch.ones((self.num_cf, 4))
                # obs.append(t)
            else:
                obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 19:28]]
        else:
            obs = [self.rpos, self.drone_state[..., 7:10], self.drone_state[..., 19:28]]

        #obs = torch.concat(obs, dim=1).unsqueeze(0)
        obs = torch.concat([o.to(self.device) for o in obs], dim=1).unsqueeze(0)
        
        state = obs.squeeze(1)

        if self.policy_select == 1:
            t = torch.tensor(self.progress_buf / self.max_episode_length, dtype=torch.float32, device=self.device)
            self.time_encoding_dim = 4
            t = t.view(-1, 1)  # Ensure correct shape before expanding
            t = t.expand(-1, self.time_encoding_dim)
            state = torch.concat([obs, t.expand(-1, 4).unsqueeze(1)], dim=-1).squeeze(1)

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        reward = torch.zeros((self.num_envs, 1, 1))
        done = torch.zeros((self.num_envs, 1, 1)).bool()
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": done,
                "terminated": done,
                "truncated": done
            },
            self.num_envs,
        )
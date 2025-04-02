import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
import collections
from tensordict.tensordict import TensorDict, TensorDictBase
from functorch import vmap

class FakePayloadTrack(FakeEnv):
    def __init__(self, cfg, connection, swarm, dt=0.01):
        self.alpha = 0.8
        self.num_envs = 1
        self.cfg = cfg
        self.future_traj_steps = 10
        self.dt = dt
        print('dt', self.dt)
        self.num_cf = 1
        self.task = 'slow' # 'slow', 'normal', 'fast', 'debug'
        self.use_time_encoding = False
        self.use_random_init = False
        self.max_episode_length = 1000

        super().__init__(cfg, connection, swarm)

        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )

        self.traj_c_dist = D.Uniform(
            torch.tensor(-0., device=self.device),
            torch.tensor(0., device=self.device)
        )

        if self.task == 'slow':
            self.T_scale_dist = D.Uniform(
                torch.tensor(15.0, device=self.device),
                torch.tensor(15.0, device=self.device)
            ) # slow
        elif self.task == 'normal':
            self.T_scale_dist = D.Uniform(
                torch.tensor(5.5, device=self.device),
                torch.tensor(5.5, device=self.device)
            ) # normal 
        elif self.task == 'fast':
            self.T_scale_dist = D.Uniform(
                torch.tensor(3.5, device=self.device),
                torch.tensor(3.5, device=self.device)
            ) # fast
        elif self.task == 'debug':
            self.T_scale_dist = D.Uniform(
                torch.tensor(4.5, device=self.device),
                torch.tensor(4.5, device=self.device)
            ) # debug
            
        self.traj_w_dist = D.Uniform(
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.0, device=self.device)
        )

        #print("T_scale_dist init:", self.T_scale_dist)

        self.origin = torch.tensor([0., 0., 1.], device=self.device)
        self.offset = torch.tensor([0.001, 0., 0.], device=self.device)

        # self.traj_t0 = torch.ones(self.num_envs, device=self.device)
        # self.T_scale = torch.ones(self.num_envs, device=self.device)

        self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device)
        self.T_scale = torch.ones(self.num_envs, 1, device=self.device)

        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
        
        self.payload_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.drone_payload_rpos = torch.zeros(self.num_envs, 3, device=self.device)

        self.prev_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # reset / initialize
        env_ids = torch.tensor([0])
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.T_scale[env_ids] = self.T_scale_dist.sample(env_ids.shape)

        #print("T_scale sample:", self.T_scale[env_ids])

        self.traj_w[env_ids] = self.traj_w_dist.sample(env_ids.shape)
        if self.use_random_init:
            self.traj_t0[env_ids] = torch.rand(env_ids.shape).to(self.device) * self.T_scale[env_ids] # 0 ~ T
        else:
            self.traj_t0[env_ids] = 0.25 * self.T_scale[env_ids]
            # self.traj_t0[env_ids] = 0.4 * self.T_scale[env_ids]

        # add init_action to self.action_history_buffer
        # init: hover
        self.prev_actions[env_ids] = torch.tensor([[0.0000, 0.0000, 0.0000, 0.5828]]).to(self.device)
        for _ in range(self.action_history):
            self.action_history_buffer.append(self.prev_actions) # add all prev_actions, not len(env_ids)

        self.target_poses = []

    def _set_specs(self):
        self.use_action_history = False # def False
        self.action_history_step = 1
        self.use_obs_norm = False

        drone_state_dim = 3 + 3 + 3 + 3 + 3 + 3 # quat payload rpos, target payload rpos, linear vel, payload linear vel, heading, lateral, up
        observation_dim = drone_state_dim + 3 * self.future_traj_steps

        if self.use_time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
            state_dim = observation_dim
        else:
            state_dim = observation_dim + 4

        # action history
        self.action_history = self.action_history_step if self.use_action_history else 0
        self.action_history_buffer = collections.deque(maxlen=self.action_history)
        if self.action_history > 0:
            observation_dim += self.action_history * 4

        if self.use_obs_norm:
            rpos_scale = [0.1, 0.1, 0.1] * self.future_traj_steps
            vel_scale = [0.1, 0.1, 0.1]
            rotation_scale = [1.0] * 9
            self.obs_norm_scale = torch.tensor(rpos_scale + vel_scale + rotation_scale)

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "state": UnboundedContinuousTensorSpec((1, state_dim),device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        # self.state_spec = CompositeSpec({
        #     "agents": CompositeSpec({
        #         "state": UnboundedContinuousTensorSpec((1, 50),device=self.device), # add motor speed
        #     })
        # }).expand(self.num_envs).to(self.device)
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
        self.info_spec = CompositeSpec({
            "agents": CompositeSpec({
                "target_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "real_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            #state_key=("agents", "intrinsics"),
            state_key=("agents", "state"),
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.target_pos[:] = self._compute_traj(steps=self.future_traj_steps, step_size=5)
        # print(self.target_pos[:, 0])
        
        #self.rpos = self.target_pos.cpu() - self.drone_state[..., :3]
        
        self.drone_payload_rpos = self.drone_state[..., :3].cpu() - self.drone_state[..., 28:31].cpu()
        self.target_payload_rpos = self.target_pos.cpu() - self.drone_state[..., 28:31].cpu()
        
        # print('payload pos', self.drone_state[..., 28:31])
        # print('payload vel', self.drone_state[..., 31:34])

        # obs = [self.rpos.flatten(1), self.drone_state[..., 3:10], self.drone_state[..., 13:19]] # old version
        obs = [
            #self.rpos.flatten(1),
            self.drone_payload_rpos.flatten(1), # drone pos - payload pos
            self.target_payload_rpos.flatten(1), # target pos - payload pos
            self.drone_state[..., 7:10], # linear v
            self.drone_state[..., 31:34], # payload linear v
            self.drone_state[..., 19:28], # rotation
        ]

        #print('obs shape prev time enco', len(obs))
        # for tensor in obs:
        #     print(tensor.shape)
                  
        if self.use_time_encoding:
            t = torch.tensor(self.progress_buf / self.max_episode_length, dtype=torch.float32, device=self.device)
            self.time_encoding_dim = 4
            t = t.view(-1, 1)  # Ensure correct shape before expanding
            t = t.expand(-1, self.time_encoding_dim)

            # Move the tensor to the same device as the other tensors in `obs`
            obs.append(t.to(self.device))

        # Concatenate all tensors in `obs`, ensuring they are on the same device
        obs = torch.concat([o.to(self.device) for o in obs], dim=1).unsqueeze(0)
        state = obs.squeeze(1)

        #obs = torch.concat(obs, dim=1).unsqueeze(0)
        #obs = torch.cat(obs, dim=-1)
        
        # print('obs shape', obs.shape)
        # print('state shape', state.shape)

        # add time encoding
        # t = torch.tensor(self.progress_buf / self.max_episode_length, dtype=torch.float32, device=self.device)
        # self.time_encoding_dim = 4
        # t = t.view(-1, 1)  # Ensure correct shape before expanding
        # t = t.expand(-1, self.time_encoding_dim)
        # state.append(t.to(self.device))

        if self.use_time_encoding:
            state = obs.squeeze(1)
        else:
            t = torch.tensor(self.progress_buf / self.max_episode_length, dtype=torch.float32, device=self.device)
            self.time_encoding_dim = 4
            t = t.view(-1, 1)  # Ensure correct shape before expanding
            t = t.expand(-1, self.time_encoding_dim)
            state = torch.concat([obs, t.expand(-1, 4).unsqueeze(1)], dim=-1).squeeze(1)

        if self.use_obs_norm:
            obs = obs * self.obs_norm_scale.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1, 1)

        #print('add time encoding state shape', state.shape)
        
        # add action history to actor
        if self.action_history > 0:
            self.action_history_buffer.append(self.prev_actions)
            all_action_history = torch.concat(list(self.action_history_buffer), dim=-1).unsqueeze(1).cpu()
            obs = torch.concat([obs, all_action_history], dim=-1)
        # print('prev_actions', all_action_history)

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
                "target_position": self.target_pos[..., 0, :],
                "real_position": self.drone_state[..., :3],
                "payload_position": self.payload_pos[..., :3],
                "drone_state": self.drone_state,
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        
        #distance = torch.norm(self.rpos[:, [0]][:2], dim=-1)
        distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        
        # reward = torch.zeros((self.num_envs, 1, 1))
        # reward[..., 0] = distance.mean()
        reward = distance
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

    def _compute_traj(self, steps: int, env_ids=torch.tensor([0]), step_size: float=1.):
        t = self.progress_buf + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + t * self.dt * torch.ones(self.num_envs, 1, device=self.device)
        #print('t shape', t.shape)
        #print('self.T_scale shape', self.T_scale.shape)
        
        #target_pos = vmap(lemniscate_v)(t, self.T_scale[env_ids].unsqueeze(-1))
        #print("t value:", t)

        target_pos = vmap(lemniscate_v)(t, self.T_scale.expand(-1, 10))

        #print("target_pos:", target_pos)

        return self.origin + target_pos

    def save_target_traj(self, name):
        torch.save(self.target_poses, name)

def pentagram(t):
    x = -1.0 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    y = 1.0 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    # x = -1.1 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    # y = 1.1 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    z = torch.zeros_like(t)
    return torch.stack([x,y,z], dim=-1)

def lemniscate_v(t, T):
    sin_t = torch.sin(2 * torch.pi * t / T)
    cos_t = torch.cos(2 * torch.pi * t / T)

    x = torch.stack([
        cos_t, sin_t * cos_t, torch.zeros_like(t)
    ], dim=-1)

    return x

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def circle(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    x = torch.stack([
        cos_t, sin_t, torch.zeros_like(sin_t)
    ], dim=-1)

    return x

def square(t_s):
    x_s = []
    for t_ in t_s[0]:
        t = torch.abs(t_).item()
        while t >= 8:
            t -= 8
        if t < 2:
            x = torch.tensor([-1., 1-t, 0.])
        elif t < 4:
            x = torch.tensor([t-3, -1., 0.])
        elif t < 6:
            x = torch.tensor([1., t-5, 0.])
        elif t < 8:
            x = torch.tensor([7-t, 1., 0.])
        x_s.append(x)
    x_s = torch.stack(x_s, dim=0).unsqueeze(0).to(t_s.device)
    return x_s

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))

import functools
def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg 
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped

@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c
# crazyflie 2.1 current weight is 35g.

import logging
import os
import sys
import time
import hydra
import torch
import numpy as np
from functorch import vmap

from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH #, init_simulation_app

from torchrl.collectors import SyncDataCollector 
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    VelController,
    AttitudeController,
    RateController,
    History
)
#from omni_drones.learning.ppo import PPORNNPolicy, PPOPolicy

from omni_drones.learning import (
    MAPPOPolicy, 
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)

#from tqdm import tqdm

from fake import FakeHover, FakeTrack, SwarmPayload, FakePayloadTrack
import time

from crazyflie_py import Crazyswarm
from torchrl.envs.utils import step_mdp
import collections

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy")
def main(cfg):
    print("Starting...")
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    print("torch seed...")
    algos = {
        "mappo": MAPPOPolicy, 
    }

    swarm = SwarmPayload(cfg, test=False)
    print("Create swarm...")
    
    cmd_fre = 100
    rpy_scale = 180
    min_thrust = 0.0
    max_thrust = 0.9
    use_track = True

    # load takeoff checkpoint
    print("takeoff loading waiting...")
    #takeoff_ckpt = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/hover.pt"
    takeoff_ckpt = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/hover_train_245M_35g_obs_15.pt"
    takeoff_env = FakeHover(cfg, connection=True, swarm=swarm)

    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    print("Successfully Takeoff load model!")
    
    #ckpt_name = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/deploy.pt"
    
    #ckpt_name = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/track_train.pt"
    
    ckpt_name = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/payload_train.pt"
    
    #ckpt_name = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/track_train_491M_mass_35g_smo_04.pt"
    #ckpt_name = "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/track_train_491M_mass_35g_smo_2.pt"

    base_env = env = FakePayloadTrack(cfg, connection=True, swarm=swarm, dt=1.0 / cmd_fre)

    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    
    # print("state_dict", state_dict.keys())
    # print("state_dict", state_dict["actor_params"])
    #print("state_dict", state_dict["critic"])

    policy.load_state_dict(state_dict)
    print("Successfully Policy load model!")
    
    #print(policy.agent_spec.observation_spec)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = base_env.reset().to(dest=base_env.device)
        print("base env reset")       

        data = policy(data, deterministic=True)
        print("policy inference")

        data = takeoff_env.reset().to(dest=takeoff_env.device)

        print("takeoff env reset")
        data = takeoff_policy(data, deterministic=True)

        print("Initializing swarm...")
        swarm.init()
        print("Swarm initialized...")

        last_time = time.time()
        data_frame = []

        # update observation
        target_pos = takeoff_env.drone_state[..., :3]
        print("current drone position:", target_pos)

        takeoff_env.target_pos = torch.tensor([[0.0, 0.0, 1.25]]) # 0.25T
        
        takeoff_frame = []
        # takeoff
        print('takeoff start')

        for timestep in range(1000):
            #print('start pos', takeoff_env.drone_state[..., :3])

            data = takeoff_env.step(data)
            data = step_mdp(data)
            
            data = takeoff_policy(data, deterministic=True)
            takeoff_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=180, rate=100)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        
        print('start eval policy position:', takeoff_env.drone_state[..., :3])
        
        data = base_env.reset().to(dest=base_env.device)          
        data = policy(data, deterministic=True)
        print("data reset")

        # real policy rollout
        print('real policy start')

        if use_track:
            #def 3500
            for track_step in range(1550):
                data = base_env.step(data) 
                data = step_mdp(data)
                
                data = policy(data, deterministic=True)
                data_frame.append(data.clone())
                action = torch.tanh(data[("agents", "action")])
                
                swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre, min_thrust=min_thrust, max_thrust=max_thrust)
                
                # setup prev_action, for prev_actions in obs
                target_rpy, target_thrust = action[:, 0, 0:3], action[:, 0, 3:]
                target_thrust = torch.clamp((target_thrust + 1) / 2, min=min_thrust, max=max_thrust)
                base_env.prev_actions = torch.concat([target_rpy, target_thrust], dim=-1)

                cur_time = time.time()
                dt = cur_time - last_time
                # print('time', dt)
                last_time = cur_time
            print('real policy done')

        
        #takeoff_env.target_pos = torch.tensor([[0., 0., 1.0]])      
    
        print('target', takeoff_env.target_pos)
        
        data = takeoff_env.reset().to(dest=takeoff_env.device)        
        data = takeoff_policy(data, deterministic=True)
        
        #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 1.0]])
        takeoff_env.target_pos = torch.tensor([[0., 0., 1.2]])   
        # print("takeoff data env reset")

        # land
        # swarm.land()

        print('land start')
        for timestep in range(1600):            
            
            ##swarm.land()
            data = takeoff_env.step(data)
            data = step_mdp(data)

            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=180, rate=100)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 100:
                target_pos[..., 2] = 1.0
                takeoff_env.target_pos = torch.tensor([[0., 0., 1.2]])
                #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 1.0]])

            if timestep == 600:
                target_pos[..., 2] = 0.8
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.7]])
                #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 0.8]])

            if timestep == 750:
                target_pos[..., 2] = 0.6
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.6]])
                #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 0.6]])

            if timestep == 900:
                target_pos[..., 2] = 0.4
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.4]])
                #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 0.4]])

            if timestep == 1050:
                target_pos[..., 2] = 0.2
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.2]])
                #takeoff_env.target_pos = torch.tensor([[base_env.drone_state[..., 0], base_env.drone_state[..., 1], 0.2]])
        
        print('land pos', takeoff_env.drone_state[..., :3])

    swarm.end_program()
    torch.save(data_frame, "/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/data/arena_slow_lemni_hover_payload.pt")


if __name__ == "__main__":
    main()
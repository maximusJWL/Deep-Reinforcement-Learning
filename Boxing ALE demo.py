import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args

parser = add_rllib_example_script_args(
    default_reward=float("inf"),
    default_timesteps=10_000_000,
    default_iters=10_000_000_000,
)
parser.set_defaults(
    enable_new_api_stack=True,
    env="ale_py:ALE/Boxing-v5",
)
parser.add_argument("--track-progress", action="store_true", default=True)
args = parser.parse_args([])

ENV = args.env
checkpoint_path = None #add path to algorithm checkpoint here


def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(ENV, **cfg, render_mode="human"),
        framestack=None,
    )

tune.register_env("env", _env_creator)

if not ray.is_initialized():
    ray.init()

config = (
    PPOConfig()
    .environment(
        "env",
        env_config={
            "frameskip": 4,
            "full_action_space": False,
            "repeat_action_probability": 0,
        },
        clip_rewards=False,
    )
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        rollout_fragment_length="auto",
        batch_mode="truncate_episodes",
        num_cpus_per_env_runner=1,
    )
    .resources(
        num_cpus_for_local_worker=1,
        num_cpus_per_worker=1,
    )
    .training(
        train_batch_size_per_learner=2048,
        minibatch_size=64,
        lambda_=0.95,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        num_epochs=2,
        lr=3e-4,
        grad_clip=100.0,
        grad_clip_by="global_norm",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_filters=[[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
            conv_activation="relu",
            head_fcnet_hiddens=[256],
            vf_share_layers=True,
        ),
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_num_workers=1,
        evaluation_config={"render_env": True},
    )
    .framework("torch")
)

tune.run(
    "PPO",
    config=config.to_dict(),     
    restore=checkpoint_path,
    stop={"training_iteration": 1},
)

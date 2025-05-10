import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
import pandas as pd

parser = add_rllib_example_script_args(
    default_reward=float("inf"),
    default_timesteps=10000000,
    default_iters=10000000000,
)
parser.set_defaults(
    enable_new_api_stack=True,
    env="ale_py:ALE/Boxing-v5",
)

parser.add_argument("--track-progress", action="store_true", default=True)
args = parser.parse_args([])
ENV = args.env

if not ray.is_initialized():
    ray.init(num_cpus=1) # scale to your machine, I ran with 14 here

def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(ENV, **cfg),# render_mode="human"),
        framestack=None,
    )

tune.register_env("env", _env_creator)

config = (
    PPOConfig()
    .environment(
        "env",
        env_config={
            "frameskip": 4,
            "full_action_space": False,
            "repeat_action_probability": 0.05,
        },
        clip_rewards=True,
    )
    .env_runners(
        num_env_runners=1, #scale to your machine, I ran with 12 here
        num_envs_per_env_runner=1,
        rollout_fragment_length='auto',
        batch_mode="truncate_episodes",
        num_cpus_per_env_runner=0.9,
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
        lr=0.00015,
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
    .reporting(
        metrics_num_episodes_for_smoothing=10,
        min_sample_timesteps_per_iteration=1000,
        keep_per_episode_custom_metrics=True,
    )
    .framework("torch")
)

if __name__ == "__main__":
    algo = config.build_algo()

    all_results = []

    print("\n Training in Progress")
    last_reported = 0

    for i in range(10000): # will take a very long time to run without large ray cluster (14 CPUs takes ~9h)
        result = algo.train()

        # pull the env-runner metrics dict
        runner = result[ENV_RUNNER_RESULTS]

        # weird quirk when using different cluster configs, key names are sometimes different
        if "episode_return_mean" in runner:
            mean_k, max_k, min_k, len_k = (
                "episode_return_mean",
                "episode_return_max",
                "episode_return_min",
                "episode_len_mean",
            )
        else:
            mean_k, max_k, min_k, len_k = (
                "episode_reward_mean",
                "episode_reward_max",
                "episode_reward_min",
                "episode_len_mean",
            )

        # summary dict
        summary = {
            "iteration":            i,
            "episode_return_mean":  runner.get(mean_k, 0),
            "episode_return_max":   runner.get(max_k, 0),
            "episode_return_min":   runner.get(min_k, 0),
            "episode_len_mean":     runner.get(len_k, 0),
            "timesteps_total":      result.get("num_env_steps_sampled_lifetime", 0),
            "time_since_start":     result.get("time_since_restore", 0),
            "training_iteration":   result.get("training_iteration", 0),
        }

        # flatten the full result and merge
        flat_full = pd.json_normalize(result, sep="_").iloc[0].to_dict()
        merged   = {**flat_full, **summary}
        all_results.append(merged)

        # periodic printing + checkpointing
        if i % 10 == 0:
            mean_r = summary["episode_return_mean"]
            if last_reported is None:
                delta = 0
            else:
                diff  = mean_r - last_reported
                delta = f"{'+' if diff>=0 else ''}{diff:.2f}"
            last_reported = mean_r

            print(
                f"Iter {i:4d} | "
                f"Mean: {mean_r:6.2f} | Delta: {delta:>6} | " #delta is diff between current and previous mean reward
                f"Steps: {summary['timesteps_total']:8d} | "
                f"Max: {summary['episode_return_max']:5.1f} | "
                f"LenAvg: {summary['episode_len_mean']:6.1f} |"
                f" Time: {summary['time_since_start']:6.1f}s | "
            )

    # after training: write out CSV & final checkpoint
    df = pd.DataFrame(all_results)
    df.to_csv("metrics.csv", index=False)
    print("All metrics saved to metrics.csv")

    final_ckpt = algo.save()
    print(f"Final model saved at: {final_ckpt}")

    ray.shutdown()
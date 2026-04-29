"""
train.py — Example training script for PlanarRobot3DOFEnv using SAC / TD3 / PPO
================================================================================
Usage:
    python train.py --algo sac --timesteps 500000
"""

import argparse
from envs import PlanarRobot3DOFEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Import stable-baselines3 algorithms
    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.env_checker import check_env

    render_mode = "human" if args.render else None
    env = PlanarRobot3DOFEnv(render_mode=render_mode)

    print("✅ Validating environment...")
    check_env(env)
    print("✅ Environment valid!\n")

    ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}
    AlgoClass = ALGOS[args.algo]

    policy = "MlpPolicy"
    model = AlgoClass(
        policy, env,
        verbose=1,
        tensorboard_log=f"./tb_logs/{args.algo}/",
    )

    print(f"🚀 Training {args.algo.upper()} for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(f"models/planar_robot_{args.algo}")
    print(f"💾 Model saved to models/planar_robot_{args.algo}.zip")

    env.close()

if __name__ == "__main__":
    main()

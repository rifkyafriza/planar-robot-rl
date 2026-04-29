"""
evaluate.py — Evaluasi dan visualisasi model yang sudah ditraining
==================================================================
Jalankan:
    python evaluate.py --model models/planar_robot_sac --episodes 10 --render
"""

import argparse
import numpy as np
from envs import PlanarRobot3DOFEnv


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    from stable_baselines3 import SAC, TD3, PPO

    render_mode = "human" if render else None
    env = PlanarRobot3DOFEnv(render_mode=render_mode)

    # Auto-detect algorithm from filename
    algo_map = {"sac": SAC, "td3": TD3, "ppo": PPO}
    algo_name = next((k for k in algo_map if k in model_path.lower()), "sac")
    model = algo_map[algo_name].load(model_path, env=env)

    print(f"🤖 Model  : {model_path}")
    print(f"🔧 Algo   : {algo_name.upper()}")
    print(f"📊 Episodes: {n_episodes}\n")

    successes, rewards, steps_list = 0, [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, step = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if terminated or truncated:
                break
        success = info.get("distance_to_target", 1.0) < env.goal_threshold
        successes += int(success)
        rewards.append(total_reward)
        steps_list.append(step)
        status = "✅" if success else "❌"
        print(f"  Ep {ep+1:02d}: reward={total_reward:7.2f}  steps={step:4d}  dist={info.get('distance_to_target', -1):.3f}m  {status}")

    print(f"\n{'─'*50}")
    print(f"  Success rate : {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Avg reward   : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg steps    : {np.mean(steps_list):.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.render)

#!/usr/bin/env python3
"""Train PPO agent on Gymnasium environments with recording capabilities"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import json
import argparse
from typing import Optional
import cv2

# Import gymnasium-robotics to register MuJoCo and other robotics environments
try:
    import gymnasium_robotics
    print("Successfully imported gymnasium-robotics environments")
except ImportError:
    print("Warning: gymnasium-robotics not installed. MuJoCo environments unavailable.")


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback based on improvement threshold and patience"""
    def __init__(self, patience=10, min_improvement=0.01, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.eval_rewards = []
    
    def _on_step(self) -> bool:
        return True
    
    def check_improvement(self, mean_reward):
        """Check if there's sufficient improvement"""
        if mean_reward > self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.2f}")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"No improvement for {self.no_improvement_count} evaluations")
        
        # Trigger early stopping if patience exceeded
        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.no_improvement_count} evaluations")
            return False
        return True


def make_gym_env(env_id, rank=0, seed=0):
    """Create a Gymnasium environment suitable for training"""
    def _init():
        # Try to create the environment, checking for version compatibility
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except gym.error.DeprecatedEnv:
            # Try to use the latest version if the requested version is deprecated
            print(f"Warning: {env_id} is deprecated. Trying latest version...")
            # Extract base env name and try with v5 (latest for most MuJoCo envs)
            base_name = env_id.rsplit('-', 1)[0]
            try:
                env = gym.make(f"{base_name}-v5", render_mode="rgb_array")
                print(f"Using {base_name}-v5 instead")
            except:
                # Fall back to v4 if v5 doesn't exist
                try:
                    env = gym.make(f"{base_name}-v4", render_mode="rgb_array")
                    print(f"Using {base_name}-v4 instead")
                except:
                    # If all else fails, use the original
                    env = gym.make(env_id, render_mode="rgb_array")
        
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(env_id="CartPole-v1", total_timesteps=100000, n_envs=4,
              reward_threshold=None, early_stopping_patience=10, min_improvement=0.01):
    """Train PPO agent with early stopping mechanism"""
    print(f"Training PPO on {env_id} with early stopping")
    print(f"Early stopping patience: {early_stopping_patience}, Min improvement: {min_improvement}")
    
    # Create vectorized environment using our custom env creator
    env = make_vec_env(
        make_gym_env(env_id, rank=0, seed=42),
        n_envs=n_envs,
        seed=42
    )
    
    # Create eval environment with version handling
    try:
        eval_env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.DeprecatedEnv:
        base_name = env_id.rsplit('-', 1)[0]
        try:
            eval_env = gym.make(f"{base_name}-v5", render_mode="rgb_array")
            print(f"Using {base_name}-v5 for evaluation")
        except:
            try:
                eval_env = gym.make(f"{base_name}-v4", render_mode="rgb_array")
                print(f"Using {base_name}-v4 for evaluation")
            except:
                eval_env = gym.make(env_id, render_mode="rgb_array")
    
    # Define model with appropriate policy based on observation space
    obs_space = env.observation_space
    if len(obs_space.shape) == 3:  # Image observations
        policy = "CnnPolicy"
    else:  # Vector observations
        policy = "MlpPolicy"
    
    model = PPO(
        policy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/"
    )
    
    # Create callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        patience=early_stopping_patience,
        min_improvement=min_improvement,
        verbose=1
    )
    
    # Modified eval callback with early stopping
    class EvalWithEarlyStopping(EvalCallback):
        def __init__(self, early_stop_callback, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.early_stop = early_stop_callback
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                if len(self.evaluations_results) > 0:
                    last_mean_reward = self.evaluations_results[-1][0]
                    should_continue = self.early_stop.check_improvement(last_mean_reward)
                    if not should_continue:
                        return False
            
            return result
    
    eval_callback = EvalWithEarlyStopping(
        early_stopping,
        eval_env,
        best_model_save_path=f"./models/ppo_{env_id}/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Reward threshold stopping
    if reward_threshold is not None:
        reward_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=1
        )
        callbacks.append(EvalCallback(
            eval_env,
            callback_on_new_best=reward_callback,
            eval_freq=5000,
            verbose=0
        ))
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix=f"ppo_{env_id}"
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            tb_log_name=env_id
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    model.save(f"models/ppo_{env_id}_final")
    print(f"Model saved to models/ppo_{env_id}_final")
    
    return model


def test_agent(model_path, env_id, episodes=5, render=True):
    """Test a trained PPO agent"""
    print(f"\nTesting PPO agent on {env_id}")
    
    # Always use rgb_array to avoid GUI dependencies
    # The render parameter now only controls whether we display stats
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.DeprecatedEnv:
        base_name = env_id.rsplit('-', 1)[0]
        try:
            env = gym.make(f"{base_name}-v5", render_mode="rgb_array")
            print(f"Using {base_name}-v5 for testing")
        except:
            try:
                env = gym.make(f"{base_name}-v4", render_mode="rgb_array")
                print(f"Using {base_name}-v4 for testing")
            except:
                env = gym.make(env_id, render_mode="rgb_array")
    
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print(f"\nStatistics over {episodes} episodes:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    
    env.close()


def record_video_with_actions(
    model_path: str,
    env_id: str,
    num_episodes: int = 3,
    record_actions: bool = True,
    actions_format: str = "csv",
    video_dir: str = "./videos",
):
    """Record video of trained agent with action logging.
    
    Saves videos and optionally logs per-step actions with frame references.
    Adapted for standard Gymnasium environments.
    """
    print(f"Recording videos for {env_id}")
    
    os.makedirs(video_dir, exist_ok=True)
    
    # Load model
    model = PPO.load(model_path)
    
    # Get action meanings if available
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.DeprecatedEnv:
        base_name = env_id.rsplit('-', 1)[0]
        try:
            env = gym.make(f"{base_name}-v5", render_mode="rgb_array")
            print(f"Using {base_name}-v5 for recording")
        except:
            try:
                env = gym.make(f"{base_name}-v4", render_mode="rgb_array")
                print(f"Using {base_name}-v4 for recording")
            except:
                env = gym.make(env_id, render_mode="rgb_array")
    
    action_meanings = None
    if hasattr(env, 'get_action_meanings'):
        action_meanings = env.get_action_meanings()
    elif hasattr(env.unwrapped, 'get_action_meanings'):
        action_meanings = env.unwrapped.get_action_meanings()
    
    # Record episodes
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + episode)
        
        video_name = os.path.join(video_dir, f"{env_id}_ep{episode+1}.mp4")
        fps = 30
        frames = []
        
        # Action logging setup
        actions_data = []
        actions_file = None
        if record_actions:
            actions_base = os.path.join(video_dir, f"{env_id}_actions_ep{episode+1}")
            if actions_format == "jsonl":
                actions_file = actions_base + ".jsonl"
            else:
                actions_file = actions_base + ".csv"
        
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 1000:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get frame
            frame = env.render()
            
            # Add info overlay to frame
            height, width = frame.shape[:2]
            overlay = frame.copy()
            
            # Add text info
            cv2.putText(overlay, f"Step: {step_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Action: {action}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Reward: {reward:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Total: {total_reward:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            frames.append(overlay)
            
            # Log action
            if record_actions:
                action_id = int(action) if hasattr(action, "__int__") else int(np.array(action).item())
                action_name = action_meanings[action_id] if action_meanings else str(action_id)
                
                action_record = {
                    "step": step_count,
                    "action_id": action_id,
                    "action_name": action_name,
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                actions_data.append(action_record)
            
            total_reward += reward
            step_count += 1
        
        # Write video
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Episode {episode + 1}: Saved {video_name}")
        
        # Save actions
        if record_actions and actions_file:
            if actions_format == "jsonl":
                with open(actions_file, 'w') as f:
                    for record in actions_data:
                        f.write(json.dumps(record) + '\n')
            else:  # CSV
                with open(actions_file, 'w') as f:
                    f.write("step,action_id,action_name,reward,total_reward,terminated,truncated\n")
                    for record in actions_data:
                        f.write(f"{record['step']},{record['action_id']},{record['action_name']},"
                               f"{record['reward']},{record['total_reward']},"
                               f"{int(record['terminated'])},{int(record['truncated'])}\n")
            print(f"Episode {episode + 1}: Saved actions to {actions_file}")
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    
    env.close()
    print(f"Videos saved in {video_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train/Test PPO on Gymnasium environments")
    parser.add_argument("--env", default="CartPole-v1", help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Minimum improvement threshold")
    parser.add_argument("--reward-threshold", type=float, help="Stop when reward threshold reached")
    parser.add_argument("--test", action="store_true", help="Test a trained model")
    parser.add_argument("--record", action="store_true", help="Record video of agent")
    parser.add_argument("--record-actions", action="store_true", help="Save per-step actions")
    parser.add_argument("--actions-format", choices=["csv", "jsonl"], default="csv", help="Actions format")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test/record episodes")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    if args.test:
        if not args.model:
            args.model = f"models/ppo_{args.env}_final"
        test_agent(args.model, args.env, episodes=args.episodes)
    
    elif args.record:
        if not args.model:
            args.model = f"models/ppo_{args.env}_final"
        record_video_with_actions(
            args.model,
            args.env,
            num_episodes=args.episodes,
            record_actions=args.record_actions,
            actions_format=args.actions_format
        )
    
    else:
        # Train
        print(f"Training PPO on {args.env}")
        model = train_ppo(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            reward_threshold=args.reward_threshold,
            early_stopping_patience=args.patience,
            min_improvement=args.min_improvement
        )
        
        print("\nTraining complete! Testing agent...")
        test_agent(f"models/ppo_{args.env}_final", args.env, episodes=3, render=False)


if __name__ == "__main__":
    main()
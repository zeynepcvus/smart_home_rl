import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.devices import Device, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS
from environment.slots import SlotManager
from environment.smart_home_env import SmartHomeEnv
from agents.rule_based_agent import RuleBasedAgent


def make_eval_env():
    """Değerlendirme ortamı oluştur"""
    slot_manager = SlotManager()

    slot_manager.add_device(Device(
        name="HVAC",
        category="D",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=True
    ))
    slot_manager.add_device(Device(
        name="Washing Machine",
        category="C",
        device_type=DEVICE_TYPE_SHIFTABLE,
        duration=2,
        deadline=22,
        comfort_sensitive=False
    ))
    slot_manager.add_device(Device(
        name="Lighting",
        category="A",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=False
    ))

    return SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        awake_start=8,
        sleep_start=23
    )


def evaluate_rl_agent(n_episodes=20):
    """Eğitilmiş RL ajanını değerlendir"""
    print("\n=== RL AGENT EVALUATION ===")

    # Modeli yükle
    env = DummyVecEnv([make_eval_env])
    env = VecNormalize.load("models/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("models/best_model/best_model", env=env)

    costs = []
    comfort_violations = []
    deadline_violations = []
    rewards = []

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        costs.append(info[0]["total_cost"])
        comfort_violations.append(info[0]["comfort_violations"])
        deadline_violations.append(info[0]["deadline_violations"])
        rewards.append(total_reward)

    print(f"Mean reward:              {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Mean cost:                {np.mean(costs):.2f} TL ± {np.std(costs):.2f}")
    print(f"Mean comfort violations:  {np.mean(comfort_violations):.1f}")
    print(f"Mean deadline violations: {np.mean(deadline_violations):.1f}")

    return np.mean(costs), np.mean(comfort_violations), np.mean(deadline_violations)


def evaluate_rule_based_agent(n_episodes=20):
    """Kural tabanlı ajanı değerlendir"""
    print("\n=== RULE-BASED AGENT EVALUATION ===")

    agent = RuleBasedAgent()

    costs = []
    comfort_violations = []
    deadline_violations = []
    rewards = []

    for episode in range(n_episodes):
        env = make_eval_env()
        obs, _ = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(
                slot_manager=env.slot_manager,
                current_hour=env.current_hour,
                indoor_temp=env.indoor_temp,
                temp_min=env.temp_min,
                temp_max=env.temp_max,
                awake_start=env.awake_start,
                sleep_start=env.sleep_start
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        costs.append(info["total_cost"])
        comfort_violations.append(env.comfort_violations)
        deadline_violations.append(env.deadline_violations)
        rewards.append(total_reward)

    print(f"Mean reward:              {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Mean cost:                {np.mean(costs):.2f} TL ± {np.std(costs):.2f}")
    print(f"Mean comfort violations:  {np.mean(comfort_violations):.1f}")
    print(f"Mean deadline violations: {np.mean(deadline_violations):.1f}")

    return np.mean(costs), np.mean(comfort_violations), np.mean(deadline_violations)


def compare():
    """İki ajanı karşılaştır"""
    rl_cost, rl_comfort, rl_deadline = evaluate_rl_agent()
    rb_cost, rb_comfort, rb_deadline = evaluate_rule_based_agent()

    print("\n" + "="*55)
    print("COMPARISON RESULTS (average over 20 episodes)")
    print("="*55)
    print(f"{'Metric':<25} {'RL Agent':>12} {'Rule-Based':>15}")
    print("-"*55)
    print(f"{'Cost (TL)':<25} {rl_cost:>12.2f} {rb_cost:>15.2f}")
    print(f"{'Comfort violations':<25} {rl_comfort:>12.1f} {rb_comfort:>15.1f}")
    print(f"{'Deadline violations':<25} {rl_deadline:>12.1f} {rb_deadline:>15.1f}")
    print("="*55)

    if rl_cost < rb_cost:
        print(f"\nRL agent reduced cost by {((rb_cost - rl_cost) / rb_cost * 100):.1f}%")
    else:
        print(f"\nRule-based agent has {((rl_cost - rb_cost) / rb_cost * 100):.1f}% lower cost")


if __name__ == "__main__":
    compare()
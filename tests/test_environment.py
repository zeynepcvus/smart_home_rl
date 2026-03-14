from environment.smart_home_env import SmartHomeEnv
from agents.rule_based_agent import RuleBasedAgent

def test_basic():
    """Ortamın temel işlevlerini test et"""
    env = SmartHomeEnv()

    # reset() çalışıyor mu?
    obs, info = env.reset()
    print(f"Initial observation: {obs}")        # yeni gün başlat, ilk observation'ı al.
    assert obs.shape == (8,), "Observation shape is wrong!"     #assert-> bu doğru olmalı, değilse dur ve hata ver. (observation 8 sayılık liste mi onu kontrol eder)

    # step() çalışıyor mu?
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)         # Rastgele bir karar al, ortama uygula


def test_full_episode():
    """Bir günlük simülasyonu baştan sona koştur"""
    env = SmartHomeEnv()
    agent = RuleBasedAgent()

    obs, _ = env.reset()
    total_reward = 0.0

    print("\n=== FULL EPISODE TEST (Rule-Based Agent) ===")
    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Hour {env.current_hour - 1:02d}:00 | "
              f"Cost: {info['cost']:.2f}TL | "
              f"Indoor temp: {info['indoor_temp']:.1f}°C | "
              f"Comfort: {'✓' if info['comfort_violations'] == 0 else '✗'}")

        if terminated or truncated:
            break

    print(f"\n--- RESULTS ---")
    print(f"Total reward:        {total_reward:.3f}")
    print(f"Total cost:          {info['total_cost']:.2f} TL")
    print(f"Comfort violations:  {env.comfort_violations}")
    print(f"Manual interventions:{env.manual_interventions}")

if __name__ == "__main__":
    test_basic()
    test_full_episode()
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.devices import Device, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS
from environment.slots import SlotManager
from environment.smart_home_env import SmartHomeEnv
from environment.pricing import HOURLY_PRICES


def make_env():
    """Analiz için ortam oluştur"""
    slot_manager = SlotManager()
    slot_manager.add_device(Device(
        name="HVAC", category="D",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=True
    ))
    slot_manager.add_device(Device(
        name="Washing Machine", category="C",
        device_type=DEVICE_TYPE_SHIFTABLE,
        duration=2, deadline=22,
        comfort_sensitive=False
    ))
    slot_manager.add_device(Device(
        name="Lighting", category="A",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=False
    ))
    return SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0, temp_max=24.0,
        awake_start=8, sleep_start=23
    )


def analyze():
    """Ajanın saatlik kararlarını analiz et"""

    # Modeli yükle
    env = DummyVecEnv([make_env])
    env = VecNormalize.load("models/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    model = PPO.load("models/best_model/best_model", env=env)

    print("\n=== SAATLIK KARAR ANALİZİ (10 episode ortalaması) ===\n")

    # Her saat için istatistik topla
    hvac_on_count = np.zeros(24)
    indoor_temps = np.zeros(24)
    episode_count = 10

    for _ in range(episode_count):
        obs = env.reset()
        done = False
        hour = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)

            if hour < 24:
                # HVAC kararı (slot 0)
                hvac_on_count[hour] += action[0][0]
                indoor_temps[hour] += info[0]["indoor_temp"]
            hour += 1

    hvac_on_rate = hvac_on_count / episode_count
    avg_temps = indoor_temps / episode_count

    print(f"{'Saat':<6} {'Fiyat':>8} {'Klima Açık':>12} {'İç Sıcaklık':>14}")
    print("-" * 44)
    for h in range(24):
        price = HOURLY_PRICES[h]
        price_tag = "PAHALIYSA" if price >= 2.5 else ("NORMAL" if price >= 1.2 else "UCUZ")
        print(f"{h:02d}:00  "
              f"{price:>5.1f} TL  "
              f"{'AÇIK' if hvac_on_rate[h] > 0.5 else 'KAPALI':>8}  "
              f"({hvac_on_rate[h]*100:>4.0f}%)  "
              f"{avg_temps[h]:>6.1f}°C  "
              f"{price_tag}")


if __name__ == "__main__":
    analyze()
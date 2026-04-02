import numpy as np
from environment.devices import Device, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS
from environment.slots import SlotManager
from environment.smart_home_env import SmartHomeEnv
from agents.rule_based_agent import RuleBasedAgent


def create_test_environment():
    """Test için örnek bir ev oluştur"""
    slot_manager = SlotManager()

    # Ertelenebilir cihaz: çamaşır makinesi
    washer = Device(
        name="Washing Machine",
        category="C",
        device_type=DEVICE_TYPE_SHIFTABLE,
        duration=2,
        deadline=22,
        comfort_sensitive=False
    )

    # Sürekli cihaz: klima
    hvac = Device(
        name="HVAC",
        category="D",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=True
    )

    # Sürekli cihaz: aydınlatma
    lighting = Device(
        name="Lighting",
        category="A",
        device_type=DEVICE_TYPE_CONTINUOUS,
        comfort_sensitive=False
    )

    slot_manager.add_device(washer)
    slot_manager.add_device(hvac)
    slot_manager.add_device(lighting)

    env = SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        awake_start=8,
        sleep_start=23
    )

    return env


def test_basic():
    """Temel işlevleri test et"""
    env = create_test_environment()

    # reset() çalışıyor mu?
    obs, info = env.reset()
    print(f"Başlangıç gözlemi shape: {obs.shape}")
    assert obs.shape == (34,), "Observation shape yanlış!"
    assert np.all((obs >= 0.0) & (obs <= 1.0)), "Observation 0-1 dışına çıktı!"
    print("✓ reset() çalışıyor")

    # step() çalışıyor mu?
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (34,), "Step sonrası observation shape yanlış!"
    assert np.all((obs >= 0.0) & (obs <= 1.0)), "Step sonrası observation 0-1 dışına çıktı!"
    print(f"✓ step() çalışıyor — ödül: {reward:.3f}")


def test_full_episode():
    """Tam günlük simülasyonu kural tabanlı ajanla koştur"""
    env = create_test_environment()
    agent = RuleBasedAgent()

    obs, _ = env.reset()
    total_reward = 0.0

    print("\n=== TAM EPİSODE TESTİ (Kural Tabanlı Ajan) ===")
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

        print(f"Saat {env.current_hour-1:02d}:00 | "
              f"Maliyet: {info['cost']:.3f} TL | "
              f"İç sıcaklık: {info['indoor_temp']:.1f}°C | "
              f"Konfor ihlali: {info['comfort_violations']}")

        if terminated or truncated:
            break

    print(f"\n--- SONUÇLAR ---")
    print(f"Toplam ödül:         {total_reward:.3f}")
    print(f"Toplam maliyet:      {info['total_cost']:.2f} TL")
    print(f"Konfor ihlali:       {env.comfort_violations}")
    print(f"Deadline ihlali:     {env.deadline_violations}")

    # Son kontroller
    assert env.current_hour == 24, "24 saat olmalıydı!"
    assert info["total_cost"] >= 0, "Toplam maliyet negatif olamaz!"
    assert env.comfort_violations >= 0, "Konfor ihlali negatif olamaz!"
    assert env.deadline_violations >= 0, "Deadline ihlali negatif olamaz!"
    print("✓ Tam episode testi geçti!")


if __name__ == "__main__":
    test_basic()
    test_full_episode()
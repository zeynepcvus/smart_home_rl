from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

from environment.devices import Device, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS
from environment.slots import SlotManager
from environment.smart_home_env import SmartHomeEnv

# Modelin kaydedileceği klasörler
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env():
    """Eğitim için ortam oluştur"""
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


def train():
    """PPO ajanını eğit"""

    # Ortamı hazırla
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Değerlendirme ortamı (eğitimden ayrı)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    # Her 5000 adımda modeli değerlendir, en iyisini kaydet
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_model",
        log_path=LOGS_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        verbose=1
    )

    # Her 10000 adımda yedek al
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{MODELS_DIR}/checkpoints",
        name_prefix="ppo_smart_home",
        verbose=1
    )

    # PPO ajanı
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,          # 2048'den 512'ye düşürüldü — daha hızlı geri bildirim
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        verbose=1,
        tensorboard_log=LOGS_DIR
    )

    print("Eğitim başlıyor...")
    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save(f"{MODELS_DIR}/final_model")
    env.save(f"{MODELS_DIR}/vec_normalize.pkl")
    eval_env.save(f"{MODELS_DIR}/eval_vec_normalize.pkl")
    print("Eğitim tamamlandı! Model kaydedildi.")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
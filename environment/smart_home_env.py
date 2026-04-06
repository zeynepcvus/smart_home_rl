import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.devices import (
    DEVICE_TYPE_EMPTY,
    DEVICE_TYPE_SHIFTABLE,
    DEVICE_TYPE_CONTINUOUS,
    ENERGY_CATEGORIES,
)
from environment.slots import SlotManager, MAX_SLOTS
from environment.pricing import get_price


CATEGORY_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


class SmartHomeEnv(gym.Env):
    """
    Akıllı ev enerji yönetimi simülasyon ortamı.
    1 episode = 1 gün = 24 saat.
    Kullanıcının eklediği cihazlar SlotManager üzerinden yönetilir.
    """

    def __init__(
        self,
        slot_manager: SlotManager,
        temp_min: float = 20.0,
        temp_max: float = 24.0,
        awake_start: int = 8,
        sleep_start: int = 23,
    ):
        super().__init__()

        self.slot_manager = slot_manager

        # Kullanıcının konfor tercihleri
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.awake_start = awake_start
        self.sleep_start = sleep_start

        # Ödül ağırlıkları
        self.w_cost = 0.35
        self.w_comfort = 0.55
        self.w_autonomy = 0.10

        # Durum değişkenleri
        self.current_hour = 0
        self.outdoor_temp = 15.0
        self.indoor_temp = 20.0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0

        # Günlük sıcaklık profili için taban değerler
        self.daily_temp_base = 20.0
        self.daily_temp_amplitude = 5.0

        # Her slot için 0/1 karar
        self.action_space = spaces.MultiBinary(MAX_SLOTS)

        # 4 genel + (5 slot * 6 özellik) = 34 boyut
        low = np.zeros(34, dtype=np.float32)
        high = np.ones(34, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _is_user_awake(self) -> bool:
        """Kullanıcı şu an uyanık mı?"""
        return self.awake_start <= self.current_hour < self.sleep_start

    def _is_hvac_device(self, device) -> bool:
        """
        HVAC benzeri cihazı tanımlar.
        Şu anki sistemde comfort_sensitive=True olan continuous cihazlar HVAC sayılır.
        """
        return (
            device.device_type == DEVICE_TYPE_CONTINUOUS
            and device.comfort_sensitive
        )

    def _is_lighting_device(self, device) -> bool:
        """
        Aydınlatma benzeri cihazı tanımlar.
        Şu anki sistemde comfort_sensitive=False olan continuous cihazlar lighting kabul edilir.
        """
        return (
            device.device_type == DEVICE_TYPE_CONTINUOUS
            and not device.comfort_sensitive
        )

    def _get_outdoor_temp_for_hour(self, hour: int) -> float:
        """
        Gün içi dış sıcaklık profili.
        Öğlen daha sıcak, gece daha serin olacak şekilde hafif sinüzoidal değişim uygulanır.
        """
        angle = 2 * np.pi * ((hour - 8) / 24.0)
        temp = self.daily_temp_base + self.daily_temp_amplitude * np.sin(angle)
        return float(np.clip(temp, -10.0, 45.0))

    def reset(self, seed=None, options=None):
        """
        Yeni bir gün başlat.
        Her episode başında otomatik olarak çağrılır.
        """
        super().reset(seed=seed)

        self.current_hour = 0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0

        # Günlük sıcaklık profili için baz değerler
        self.daily_temp_base = float(np.random.uniform(10, 30))
        self.daily_temp_amplitude = float(np.random.uniform(3, 8))

        # Günün ilk saatindeki dış sıcaklık
        self.outdoor_temp = self._get_outdoor_temp_for_hour(self.current_hour)

        # İç sıcaklık dış sıcaklığa yakın başlasın
        self.indoor_temp = float(
            np.random.uniform(self.outdoor_temp - 3, self.outdoor_temp + 3)
        )
        self.indoor_temp = float(np.clip(self.indoor_temp, -10.0, 45.0))

        self.slot_manager.reset_all()

        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        Mevcut durumu 34 boyutlu normalize edilmiş numpy dizisine dönüştür.
        İlk 4 değer genel bilgiler, kalan 30 değer her slot için 6 özellik.
        """
        price = get_price(self.current_hour % 24)

        general = [
            min(self.current_hour, 23) / 23.0,
            np.clip(price / 5.0, 0.0, 1.0),
            np.clip((self.outdoor_temp + 10) / 55.0, 0.0, 1.0),
            np.clip((self.indoor_temp + 10) / 55.0, 0.0, 1.0),
        ]

        slot_features = []
        for device in self.slot_manager.slots:
            slot_features.extend(
                [
                    float(device.is_active),
                    device.device_type / 2.0,
                    (
                        CATEGORY_TO_INDEX[device.category] / 4.0
                        if device.category is not None
                        else 0.0
                    ),
                    np.clip(device.remaining_hours / 24.0, 0.0, 1.0),
                    (
                        np.clip(
                            max(device.deadline - self.current_hour, 0) / 23.0,
                            0.0,
                            1.0,
                        )
                        if device.deadline is not None
                        else 0.0
                    ),
                    float(device.comfort_sensitive),
                ]
            )

        return np.array(general + slot_features, dtype=np.float32)

    def step(self, action):
        """
        1 saat ilerlet.
        Aksiyonu uygula → simülasyonu güncelle → ödülü hesapla → yeni state döndür.
        """
        invalid_action_penalty = 0.0

        # 1) AKSİYONLARI UYGULA
        for i, device in enumerate(self.slot_manager.slots):
            # Boş slot
            if device.device_type == DEVICE_TYPE_EMPTY:
                if action[i] == 1:
                    invalid_action_penalty -= 0.02
                continue

            # Ertelenebilir cihaz
            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                if action[i] == 1:
                    if device.is_completed or device.is_active or device.duration is None:
                        invalid_action_penalty -= 0.02
                    device.start()

            # Sürekli cihaz
            elif device.device_type == DEVICE_TYPE_CONTINUOUS:
                if action[i] == 1:
                    device.turn_on()
                else:
                    device.turn_off()

        # 2) SİMÜLASYONU İLERLET
        self.slot_manager.step_all()

        # İç sıcaklığı güncelle
        hvac_on = any(self._is_hvac_device(d) and d.is_active for d in self.slot_manager.slots)

        if hvac_on:
            target = (self.temp_min + self.temp_max) / 2.0
            self.indoor_temp += (target - self.indoor_temp) * 0.3
        else:
            # Daha yavaş kayma — termal atalet
            self.indoor_temp += (self.outdoor_temp - self.indoor_temp) * 0.05

        self.indoor_temp = float(np.clip(self.indoor_temp, -10.0, 45.0))

        # 3) ÖDÜLÜ HESAPLA
        price = get_price(self.current_hour % 24)
        reward, cost, comfort_ok = self._calculate_reward(price)
        reward += invalid_action_penalty

        # 4) İSTATİSTİKLER
        self.total_cost += cost
        if not comfort_ok:
            self.comfort_violations += 1

        # 5) SONRAKİ SAATE GEÇ
        self.current_hour += 1
        terminated = self.current_hour >= 24

        # Bir sonraki state için dış sıcaklığı yeni saate göre güncelle
        self.outdoor_temp = self._get_outdoor_temp_for_hour(self.current_hour % 24)

        # Gün bittiyse deadline ihlallerini say
        if terminated:
            for device in self.slot_manager.slots:
                if (
                    device.device_type == DEVICE_TYPE_SHIFTABLE
                    and not device.is_completed
                ):
                    self.deadline_violations += 1

        observation = self._get_obs()
        info = {
            "cost": cost,
            "total_cost": self.total_cost,
            "comfort_violations": self.comfort_violations,
            "deadline_violations": self.deadline_violations,
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": self.outdoor_temp,
            "invalid_action_penalty": invalid_action_penalty,
        }

        return observation, reward, terminated, False, info

    def _calculate_reward(self, price: float):
        """
        Çok amaçlı ödül fonksiyonu.
        Döndürür: (toplam_ödül, maliyet, konfor_tamam_mı)
        """
        # 1) MALİYET
        total_power = sum(d.get_power() for d in self.slot_manager.slots)
        cost = total_power * price
        cost_reward = -cost

        # 2) KONFOR
        # Sıcaklık konforu — merkeze yakınlık ödülü
        target = (self.temp_min + self.temp_max) / 2.0

        if self.temp_min <= self.indoor_temp <= self.temp_max:
            temp_ok = True
            deviation = abs(self.indoor_temp - target)
            temp_score = 1.0 - min(deviation / 4.0, 0.5)
        else:
            temp_ok = False
            if self.indoor_temp < self.temp_min:
                deviation = self.temp_min - self.indoor_temp
            else:
                deviation = self.indoor_temp - self.temp_max
            temp_score = -(1.0 + min(deviation / 3.0, 2.0))

        # Aydınlatma konforu
        is_night = self.current_hour >= 18 or self.current_hour < 8
        user_awake = self._is_user_awake()
        light_active = any(
            self._is_lighting_device(d) and d.is_active
            for d in self.slot_manager.slots
        )
        light_ok = (not is_night) or (not user_awake) or light_active
        light_score = 0.0 if light_ok else -1.0

        comfort_ok = temp_ok and light_ok
        comfort_reward = temp_score + light_score

        # 3) OTOMASYON
        autonomy_reward = 0.0
        for device in self.slot_manager.slots:
            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                if not device.is_active and not device.is_completed:
                    autonomy_reward -= 0.1
                if (
                    device.deadline is not None
                    and not device.is_completed
                    and not device.is_active
                    and self.current_hour >= device.deadline - 3
                ):
                    autonomy_reward -= 0.5

        # 4) AÇ/KAPAT CEZASI
        switch_penalty = 0.0
        for device in self.slot_manager.slots:
            if (
                device.device_type == DEVICE_TYPE_CONTINUOUS
                and device.is_active != device.previous_active_state
            ):
                switch_penalty -= 0.05

        # AĞIRLIKLI TOPLAM
        total_reward = (
            self.w_cost * cost_reward
            + self.w_comfort * comfort_reward
            + self.w_autonomy * autonomy_reward
            + switch_penalty
        )

        return total_reward, cost, comfort_ok
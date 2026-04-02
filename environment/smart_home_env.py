import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environment.devices import (Device, DEVICE_TYPE_EMPTY,
                                  DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS,
                                  ENERGY_CATEGORIES)
from environment.slots import SlotManager, MAX_SLOTS
from environment.pricing import get_price


class SmartHomeEnv(gym.Env):
    """
    Akıllı ev enerji yönetimi simülasyon ortamı.
    1 episode = 1 gün = 24 saat.
    Kullanıcının eklediği cihazlar SlotManager üzerinden yönetilir.
    """

    def __init__(self, slot_manager: SlotManager,
                 temp_min: float = 20.0, temp_max: float = 24.0,
                 awake_start: int = 8, sleep_start: int = 23):

        super().__init__()

        # Slot yöneticisi — kullanıcının cihazları burada
        self.slot_manager = slot_manager

        # Kullanıcının konfor tercihleri
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.awake_start = awake_start   # kullanıcı kaçta uyanıyor
        self.sleep_start = sleep_start   # kullanıcı kaçta uyuyor

        # Ödül ağırlıkları (toplamı 1.0)
        self.w_cost     = 0.5  # maliyet
        self.w_comfort  = 0.4  # konfor
        self.w_autonomy = 0.1  # otomasyon

        # Durum değişkenleri
        self.current_hour = 0
        self.outdoor_temp = 15.0
        self.indoor_temp = 20.0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0

        # ── ACTION SPACE ─────────────────────────────────────────
        # Her slot için ayrı ikili karar: 0=kapat/beklet, 1=aç/başlat
        self.action_space = spaces.MultiBinary(MAX_SLOTS)

        # ── OBSERVATION SPACE ────────────────────────────────────
        # Genel bilgiler: saat, fiyat, dış sıcaklık, iç sıcaklık (4 değer)
        # Her slot için: aktif mi, tip, kategori, kalan süre,
        #                deadline'a kalan süre, konfor etkili mi (6 değer × 5 slot = 30 değer)
        # Toplam: 34 boyutlu state, tümü 0-1 arasında normalize edilmiş
        low  = np.zeros(34, dtype=np.float32)
        high = np.ones(34,  dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _is_user_awake(self) -> bool:
        """Kullanıcı şu an uyanık mı?"""
        return self.awake_start <= self.current_hour < self.sleep_start

    def reset(self, seed=None, options=None):
        """
        Yeni bir gün başlat.
        Her episode başında otomatik olarak çağrılır.
        """
        super().reset(seed=seed)

        # Sayaçları sıfırla
        self.current_hour = 0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0

        # Dış sıcaklığı rastgele belirle
        self.outdoor_temp = float(np.random.uniform(5, 35))

        # İç sıcaklık dış sıcaklığa yakın başlasın
        self.indoor_temp = float(np.random.uniform(
            self.outdoor_temp - 3, self.outdoor_temp + 3
        ))

        # Tüm cihazları sıfırla
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

        # Genel bilgiler — normalize edilmiş (0-1 arası)
        # np.clip ile 0-1 dışına taşma önlendi
        general = [
            min(self.current_hour, 23) / 23.0,
            np.clip(price / 5.0, 0.0, 1.0),
            np.clip((self.outdoor_temp + 10) / 55.0, 0.0, 1.0),
            np.clip((self.indoor_temp + 10) / 55.0, 0.0, 1.0),
        ]

        # Her slot için 6 özellik
        slot_features = []
        for device in self.slot_manager.slots:
            slot_features.extend([
                float(device.is_active),
                device.device_type / 2.0,
                (list(ENERGY_CATEGORIES.keys()).index(device.category) / 4.0)
                if device.category is not None else 0.0,
                np.clip(device.remaining_hours / 24.0, 0.0, 1.0),
                (np.clip(max(device.deadline - self.current_hour, 0) / 23.0, 0.0, 1.0))
                if device.deadline is not None else 0.0,
                float(device.comfort_sensitive),
            ])

        return np.array(general + slot_features, dtype=np.float32)

    def step(self, action):
        """
        1 saat ilerlet.
        Aksiyonu uygula → simülasyonu güncelle → ödülü hesapla → yeni state döndür.
        """
        # ── AKSİYONLARI UYGULA ──────────────────────────────────
        for i, device in enumerate(self.slot_manager.slots):

            # Boş slot — hiçbir şey yapma
            if device.device_type == DEVICE_TYPE_EMPTY:
                continue

            # Ertelenebilir cihaz
            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                if action[i] == 1:
                    device.start()

            # Sürekli cihaz
            if device.device_type == DEVICE_TYPE_CONTINUOUS:
                if action[i] == 1:
                    device.turn_on()
                else:
                    device.turn_off()

        # ── SİMÜLASYONU İLERLET ─────────────────────────────────
        # Tüm cihazları 1 saat ilerlet
        self.slot_manager.step_all()

        # İç sıcaklığı güncelle
        hvac_on = any(
            d.is_active and d.comfort_sensitive
            for d in self.slot_manager.slots
            if d.device_type == DEVICE_TYPE_CONTINUOUS
        )

        if hvac_on:
            # Klima açıksa hedef sıcaklığa doğru yaklaş
            target = (self.temp_min + self.temp_max) / 2
            self.indoor_temp += (target - self.indoor_temp) * 0.3
        else:
            # Klima kapalıysa dış sıcaklığa doğru kayma
            self.indoor_temp += (self.outdoor_temp - self.indoor_temp) * 0.1

        # ── ÖDÜLÜ HESAPLA ────────────────────────────────────────
        price = get_price(self.current_hour % 24)
        reward, cost, comfort_ok = self._calculate_reward(price)

        # İstatistikleri güncelle
        self.total_cost += cost
        if not comfort_ok:
            self.comfort_violations += 1

        # ── SONRAKİ SAATE GEÇ ────────────────────────────────────
        self.current_hour += 1
        terminated = self.current_hour >= 24

        # Gün bittiyse deadline ihlallerini say
        if terminated:
            for device in self.slot_manager.slots:
                if (device.device_type == DEVICE_TYPE_SHIFTABLE
                        and not device.is_completed):
                    self.deadline_violations += 1

        observation = self._get_obs()
        info = {
            "cost": cost,
            "total_cost": self.total_cost,
            "comfort_violations": self.comfort_violations,
            "deadline_violations": self.deadline_violations,
            "indoor_temp": self.indoor_temp,
        }

        return observation, reward, terminated, False, info

    def _calculate_reward(self, price: float):
        """
        Çok amaçlı ödül fonksiyonu.
        Döndürür: (toplam_ödül, maliyet, konfor_tamam_mı)
        """
        # ── 1. MALİYET ───────────────────────────────────────────
        total_power = sum(d.get_power() for d in self.slot_manager.slots)
        cost = total_power * price
        cost_reward = -cost

        # ── 2. KONFOR ────────────────────────────────────────────
        # Sıcaklık konfor bandında mı?
        temp_ok = self.temp_min <= self.indoor_temp <= self.temp_max

        # Aydınlatma konforu:
        # gündüzse doğal ışık yeterli
        # geceyse ve kullanıcı uyanıksa lamba açık olmalı
        # kullanıcı uyuyorsa ışık şart değil
        is_night = self.current_hour >= 18 or self.current_hour < 8
        user_awake = self._is_user_awake()
        light_active = any(
            d.is_active for d in self.slot_manager.slots
            if d.device_type == DEVICE_TYPE_CONTINUOUS
            and not d.comfort_sensitive  # klima değil, aydınlatma
        )
        light_ok = (not is_night) or (not user_awake) or light_active

        comfort_ok = temp_ok and light_ok
        comfort_reward = 1.0 if comfort_ok else -1.0

        # ── 3. OTOMASYON ─────────────────────────────────────────
        autonomy_reward = 0.0
        for device in self.slot_manager.slots:
            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                # Bekliyor ama başlatılmadıysa hafif ceza
                if not device.is_active and not device.is_completed:
                    autonomy_reward -= 0.1
                # Deadline'a 3 saat kaldı hâlâ başlamadıysa artan ceza
                if (device.deadline is not None and not device.is_completed
                        and not device.is_active
                        and self.current_hour >= device.deadline - 3):
                    autonomy_reward -= 0.3

        # ── 4. AÇ/KAPAT CEZASI ───────────────────────────────────
        switch_penalty = 0.0
        for device in self.slot_manager.slots:
            if (device.device_type == DEVICE_TYPE_CONTINUOUS
                    and device.is_active != device.previous_active_state):
                switch_penalty -= 0.05

        # ── AĞIRLIKLI TOPLAM ─────────────────────────────────────
        total_reward = (
            self.w_cost     * cost_reward    +
            self.w_comfort  * comfort_reward +
            self.w_autonomy * autonomy_reward
        ) + switch_penalty

        return total_reward, cost, comfort_ok
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environment.devices import HVACDevice, WasherDevice, LightingDevice
from environment.pricing import get_price

class SmartHomeEnv(gym.Env):
    """
    Akıllı ev enerji yönetimi simülasyon ortamı.
    1 episode = 1 gün = 24 saat
    Her adımda ajan 3 karar verir:
      - Klimayı aç/kapat
      - Çamaşır makinesini başlat/beklet
      - Işık seviyesini ayarla
    """

    def __init__(self, temp_min=20.0, temp_max=24.0):
        super().__init__()

        # Cihazları oluştur
        self.hvac = HVACDevice()
        self.washer = WasherDevice
        self.lighting = LightingDevice()

        # Kullanıcının konfor tercihleri
        self.hvac.target_temp_min = temp_min
        self.hvac.target_temp_max = temp_max

        #Ödül ağırlıkları
        self.w_cost     = 0.5   #maliyet
        self.w_comfort  = 0.4    #konfor
        self.w_autonomy = 0.1   #otomasyon

        # ── OBSERVATION SPACE (ajan ne görür?) ──
        # Ajan her adımda 8 sayı görür:
        # [saat, dış_sıcaklık, iç_sıcaklık, elektrik_fiyatı,
        # klima_açık_mı, çamaşır_çalışıyor_mu, ışık_seviyesi, çamaşır_bekliyor_mu]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 10, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([23, 45, 35, 5, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # ── ACTION SPACE (ajan ne yapabilir?) ──
        # Ajan her adımda 3 karar verir:
        # [klima: 0=kapat 1=aç,
        #  çamaşır: 0=başlatma 1=başlat,
        #  ışık: 0=kapat 1=yarım 2=tam]
        self.action_space = spaces.MultiDiscrete([2, 2, 3])  # her karar için ayrı ayrı seçenek sayısı ver (multidiscrete)
                                                             # Ajan her adımda 3 karar verecek, birinci kararda 2 seçeneği var, ikincisinde 2, üçüncüsünde 3

        # ── DURUM DEĞİŞKENLERİ ──
        # Bunlar her yeni günde sıfırlanacak
        self.current_hour = 0
        self.outdoor_temp = 15.0
        self.washer_needed = False  # bugün çamaşır yıkanacak mı?
        self.total_cost = 0.0  # günlük toplam maliyet
        self.comfort_violations = 0  # kaç kez konfor ihlali oldu
        self.manual_interventions = 0  # kaç kez müdahale gerekti

    # Yeni gün başlatma
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
        self.manual_interventions = 0

        # Dış sıcaklığı rastgele belirle (5 ile 35 derece arası)
        self.outdoor_temp = float(np.random.uniform(5, 35))

        # Bugün çamaşır yıkanacak mı? (rastgele)
        self.washer_needed = bool(np.random.random() > 0.4)

        # Cihazları sıfırla
        self.hvac.reset(initial_temp=float(np.random.uniform(18,26)))   # 18-26 arası rastgele iç sıcaklıkla başlar. ev her gün farklı sıcaklıkta olabilir.
        self.washer.reset(needed=self.washer_needed)    # üstte belirlenen washer_needed bilgisini alır
        self.lighting.reset() # sıfırlanıyor, kapalı başlıyor

        observation = self._get_observation()
        info = {}
        return observation, info
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
        self.washer = WasherDevice()
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

    # Yeni gün başlatma (reset)
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

    # Mevcut durumu (O anki simülasyonun durumunu alıp) 8 sayılık numpy dizisi olarak döndür
    def _get_observation(self):
        price = get_price(self.current_hour % 24)
        return np.array([
            self.current_hour,
            self.outdoor_temp,
            self.hvac.current_temp,
            price,
            float(self.hvac.is_on),
            float(self.washer.is_running),
            self.lighting.level,
            float(self.washer.pending),
        ], dtype=np.float32)


    def step(self, action):
        """
        1 saat ilerlet.
        Aksiyonu uygula -> simülasyonu güncelle -> ödülü hesapla -> yeni state döndür.
        """

        # Aksiyonu parçalara ayır
        hvac_action, washer_action, light_action = action

        # -- AKSİYONLARI UYGULA --
        self.hvac.is_on = bool(hvac_action)

        # Çamaşır: bekleniyorsa ve çalışmıyorsa başlat
        if washer_action == 1 and self.washer.pending and not self.washer.is_running:
            self.washer.start()

        # Işık: 0=kapat, 1=yarım, 2=tam
        light_levels = [0.0, 0.5, 1.0]
        self.lighting.set_level(light_levels[light_action])

        # ── SİMÜLASYONU İLERLET ──

        # Klima sıcaklığı güncelle
        self.hvac.update_temperature(self.outdoor_temp)

        # Çamaşırı 1 saat ilerlet
        self.washer.step()

        # ÖDÜLÜ HESAPLA
        price = get_price(self.current_hour)
        reward, cost, comfort_ok = self._calculate_reward(price)

        # İstatistikleri güncelle
        self.total_cost += cost
        if not comfort_ok:
            self.comfort_violations +=1

        # --SONRAKİ SAATE GEÇ--
        self.current_hour +=1
        terminated = self.current_hour >=24     #24.saate geldiyse gün bitti (terminated=episode bitti mi?)

        # Gün bittiyse çamaşır hâlâ bekliyor mu?
        if terminated and self.washer.pending:
            self.manual_interventions +=1

        observation = self._get_observation()
        info = {
            "cost": cost,
            "total_cost": self.total_cost,
            "comfort_violations": self.comfort_violations,
            "indoor_temp": self.hvac.current_temp,
        }

        return observation, reward, terminated, False, info


    def _calculate_reward(self, price):
        """
        Çok amaçlı ödül fonksiyonu.
        Döndürür: toplam_ödül, maliyet, konfor_tamam_mı
        """

        # ── 1. MALİYET ──
        # Bu saatte tüm cihazların harcadığı enerji × fiyat
        total_power = (
            self.hvac.get_power() +
            self.washer.get_power() +
            self.lighting.get_power()
        )
        cost = total_power * price      # TL ccinsinden maliyet
        cost_reward = -cost             # maliyet arttıkça ödül azalır

        # ── 2. KONFOR ──
        temp_ok = self.hvac.is_comfortable()
        light_ok = self.lighting.is_comfortable()
        comfort_ok = temp_ok and light_ok
        comfort_reward = 1.0 if comfort_ok else -1.0

        # ── 3. OTOMASYON ───
        # Çamaşır bekliyor ama başlatılmadıysa hafif ceza
        autonomy_reward = 0.0
        if self.washer.pending and not self.washer.is_running:
            autonomy_reward = -0.2

        # ── AĞIRLIKLI TOPLAM ──
        total_reward = (        #  %50 maliyet, %40 konfor, %10 otomasyon
            self.w_cost * cost_reward +
            self.w_comfort * comfort_reward +
            self.w_autonomy * autonomy_reward
        )

        return total_reward, cost, comfort_ok


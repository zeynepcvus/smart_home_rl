import numpy as np

# Klima / Isıtma Sistemi
class HVACDevice:


    def __init__(self):     # Bu cihaz ilk oluşturulduğunda şu değerleri kullansın:
        self.name = "HVAC"
        self.power_kw = 2.5
        self.is_on = False  # Klima başta kapalı
        self.current_temp = 20.0
        self.target_temp_min = 20.0
        self.target_temp_max = 24.0     # İç sıcaklık 20, konfor aralığı 20-24 derece


    # Her saat sonunda sıcaklığı güncelle
    def update_temperature(self, outdoor_temp: float):
        if self.is_on:
            # Klima açıksa hedef sıcaklığa doğru yaklaş
            target = (self.target_temp_min + self.target_temp_max) / 2
            self.current_temp += (target - self.current_temp) * 0.3     # Hedefe doğru %30 yaklaş her saat.
        else:
            # Klima kapalıysa dış sıcaklığa doğru kayma olur
            self.current_temp += (outdoor_temp- self.current_temp) * 0.1        # Dış sıcaklığa kayar ama çok yavaş (%10)


    # Sıcaklık konfor bandında mı?
    def is_comfortable(self) -> bool:
        return self.target_temp_min <= self.current_temp <= self.target_temp_max


    def get_power(self) -> float:
        return self.power_kw if self.is_on else 0.0     # Kaç kW tüketiyor?


    def reset(self, initial_temp: float):
        self.is_on = False
        self.current_temp = initial_temp    # Klimayı kapat ve sıcaklığı başlangıca döndür. Yeni gün başlarken kullanılır


# Çamaşır makinesi
class WasherDevice:

    def __init__(self):
        self.name = "Washer"
        self.power_kw = 1.8          # çalışırken tükettiği güç
        self.cycle_duration = 2      # kaç saat sürüyor
        self.is_running = False
        self.remaining_hours = 0     # kaç saat daha çalışacak
        self.pending = False         # bugün yıkanacak mı?

    # Makineyi başlat
    def start(self):
        self.is_running = True  # makine artık  çalışıyor
        self.remaining_hours = self.cycle_duration  # geri sayım başlıyor
        self.pending= False # bekleyen iş yok, çünkü başlattık

    # 1 saat ilerlet
    def step(self):
        """ Bu fonksiyon simülasyonda **her saat** çağrılır.
            - Makine çalışıyorsa kalan süreyi 1 azalt
            - Sıfıra ulaştıysa makineyi durdur """
        if self.is_running:
            self.remaining_hours -=1
            if self.remaining_hours <= 0:
                self.is_running = False
                self.remaining_hours = 0

    # Kaç kW üretiyor?
    def get_power(self) -> float:
        return self.power_kw if self.is_running else 0.0        # çalışıyorsa 1.8kW, çalışmıyorsa 0.0kW döner

    def reset(self, needed: bool):
        self.is_running = False
        self.remaining_hours = 0
        self.pending = needed
    # Her yeni güne geçerken çağırılır. Makineyi durdur, geri sayımı sıfırla. needed= yarın çamaşır yıkanacak mı?
    # reset(needed=True)   yarın yıkama var
    # reset(needed=False)  yarın yıkama yok


# Aydınlatma sistemi
class LightingDevice:

    def __init__(self):
        self.name = "Lighting"
        self.max_power_kw = 0.5
        self.level = 0.0             # 0.0=kapalı, 0.5=yarım, 1.0=tam
        self.preferred_min = 0.3     # kullanıcının istediği minimum parlaklık
        self.preferred_max = 1.0

    # Parlaklığı ayarla
    def set_level(self, level: float):
        self.level = max(0.0, min(1.0, level))  # Sınır koruyucu

    # Parlaklık konfor bandında mı?
    def is_comfortable(self) -> bool:
        return self.preferred_min <= self.level <= self.preferred_max

    # Kaç kW tüketiyor?
    def get_power(self) -> float:
        return self.max_power_kw * self.level

    def reset(self):
        self.level = 0.0


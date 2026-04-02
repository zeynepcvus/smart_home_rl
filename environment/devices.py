# Enerji kategorileri ve güç tüketimleri (kW)
ENERGY_CATEGORIES = {
    "A": 0.3,
    "B": 0.8,
    "C": 1.5,
    "D": 2.5,
    "E": 7.0,
}

# Cihaz tipleri
DEVICE_TYPE_EMPTY = 0        # boş slot
DEVICE_TYPE_SHIFTABLE = 1    # ertelenebilir (çamaşır, bulaşık...)
DEVICE_TYPE_CONTINUOUS = 2   # sürekli (klima, aydınlatma...)

VALID_DEVICE_TYPES = {DEVICE_TYPE_EMPTY, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS}


class Device:
    """
    Genel cihaz sınıfı.
    Kullanıcının eklediği her cihaz bu sınıftan oluşturulur.
    """

    def __init__(self, name: str, category: str, device_type: int,
                 duration: int | None = None, deadline: int | None = None,
                 comfort_sensitive: bool = False):

        # Doğrulama
        if category not in ENERGY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. "
                             f"Valid categories: {list(ENERGY_CATEGORIES.keys())}")

        if device_type not in VALID_DEVICE_TYPES:
            raise ValueError(f"Invalid device type: {device_type}. "
                             f"Valid types: {VALID_DEVICE_TYPES}")

        self.name = name
        self.category = category
        self.power_kw = ENERGY_CATEGORIES[category]
        self.device_type = device_type
        self.comfort_sensitive = comfort_sensitive

        # Sadece ertelenebilir cihazlar için anlamlı
        self.duration = duration      # kaç saat çalışacak
        self.deadline = deadline      # kaça kadar bitmeli

        # Durum değişkenleri
        self.is_active = False
        self.is_completed = False
        self.remaining_hours = 0
        self.previous_active_state = False

    @classmethod
    def empty_slot(cls):
        """Boş slot oluştur"""
        slot = cls.__new__(cls)
        slot.name = "EMPTY"
        slot.category = None
        slot.power_kw = 0.0
        slot.device_type = DEVICE_TYPE_EMPTY
        slot.comfort_sensitive = False
        slot.duration = None
        slot.deadline = None
        slot.is_active = False
        slot.is_completed = False
        slot.remaining_hours = 0
        slot.previous_active_state = False
        return slot

    def get_power(self) -> float:
        """Şu an kaç kW tüketiyor?"""
        if self.device_type == DEVICE_TYPE_EMPTY:
            return 0.0
        return self.power_kw if self.is_active else 0.0

    def start(self):
        """Ertelenebilir cihazı başlat"""
        if (
            self.device_type == DEVICE_TYPE_SHIFTABLE
            and not self.is_completed
            and not self.is_active
        ):
            self.is_active = True
            self.remaining_hours = self.duration

    def turn_on(self):
        """Sürekli cihazı aç"""
        if self.device_type == DEVICE_TYPE_CONTINUOUS:
            self.is_active = True

    def turn_off(self):
        """Sürekli cihazı kapat"""
        if self.device_type == DEVICE_TYPE_CONTINUOUS:
            self.is_active = False

    def step(self):
        """1 saat ilerlet"""
        self.previous_active_state = self.is_active

        if self.device_type == DEVICE_TYPE_SHIFTABLE and self.is_active:
            self.remaining_hours -= 1
            if self.remaining_hours <= 0:
                self.is_active = False
                self.is_completed = True
                self.remaining_hours = 0

    def reset(self):
        """Yeni gün başında sıfırla"""
        self.is_active = False
        self.is_completed = False
        self.remaining_hours = 0
        self.previous_active_state = False
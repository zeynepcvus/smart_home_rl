from environment.devices import Device, DEVICE_TYPE_EMPTY

MAX_SLOTS = 5  # maksimum cihaz sayısı

class SlotManager:
    """
    5 slotluk sabit yapıyı yöneten sınıf.
    Kullanıcının cihazlarını slotlara yerleştirir,
    boş slotları otomatik doldurur.
    """

    def __init__(self):
        # 5 slot başta hepsi boş
        self.slots: list[Device] = [Device.empty_slot() for _ in range(MAX_SLOTS)]

    def add_device(self, device: Device) -> bool:
        """
        Boş bir slota cihaz ekle.
        Başarılıysa True, slot doluysa False döner.
        """
        for i in range(MAX_SLOTS):
            if self.slots[i].device_type == DEVICE_TYPE_EMPTY:
                self.slots[i] = device
                return True
        return False  # tüm slotlar dolu

    def remove_device(self, index: int) -> bool:
        """
        Belirtilen slottaki cihazı sil, slotu boşalt.
        Başarılıysa True, geçersiz index ise False döner.
        """
        if 0 <= index < MAX_SLOTS:
            self.slots[index] = Device.empty_slot()
            return True
        return False  # geçersiz index

    def get_non_empty_devices(self) -> list[Device]:
        """Boş olmayan slotlardaki cihazları döndür"""
        return [s for s in self.slots if s.device_type != DEVICE_TYPE_EMPTY]

    def get_active_running_devices(self) -> list[Device]:
        """Şu an çalışan cihazları döndür"""
        return [s for s in self.slots if s.is_active]

    def reset_all(self):
        """Tüm cihazları yeni gün için sıfırla"""
        for device in self.slots:
            device.reset()

    def step_all(self):
        """Tüm cihazları 1 saat ilerlet"""
        for device in self.slots:
            device.step()

    def get_slot_count(self) -> int:
        """Kaç slot dolu?"""
        return sum(1 for s in self.slots if s.device_type != DEVICE_TYPE_EMPTY)
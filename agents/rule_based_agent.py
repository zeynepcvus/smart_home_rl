from environment.devices import DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS, DEVICE_TYPE_EMPTY
from environment.pricing import get_price_category
from environment.slots import SlotManager


class RuleBasedAgent:
    """
    Basit kural tabanlı kontrolcü.
    RL ajanıyla karşılaştırmak için kullanılır.

    Kurallar:
    - Ertelenebilir cihazlar: sadece ucuz saatlerde başlat,
      deadline'a 3 saat kaldıysa fiyata bakmadan başlat
    - Sürekli cihazlar: konfor bozulmuşsa aç, bozulmamışsa kapat
    - Aydınlatma: gece ve kullanıcı uyanıksa aç, değilse kapat
    - Boş slot: hiçbir şey yapma
    """

    def select_action(self, slot_manager: SlotManager,
                      current_hour: int,
                      indoor_temp: float,
                      temp_min: float,
                      temp_max: float,
                      awake_start: int = 8,
                      sleep_start: int = 23) -> list[int]:
        """
        Mevcut duruma göre 5 slotluk aksiyon listesi döndür.
        Her eleman 0 veya 1.
        """
        actions = []
        price_cat = get_price_category(current_hour)

        # Gece mi? Kullanıcı uyanık mı?
        is_night = current_hour >= 18 or current_hour < 8
        user_awake = awake_start <= current_hour < sleep_start

        for device in slot_manager.slots:

            # Boş slot — her zaman 0
            if device.device_type == DEVICE_TYPE_EMPTY:
                actions.append(0)

            # Ertelenebilir cihaz
            elif device.device_type == DEVICE_TYPE_SHIFTABLE:
                if device.is_completed or device.is_active:
                    # Tamamlandıysa veya zaten çalışıyorsa dokunma
                    actions.append(0)
                elif device.deadline is not None and current_hour >= device.deadline - 3:
                    # Deadline'a 3 saat kaldı — fiyata bakmadan başlat
                    actions.append(1)
                elif price_cat == "cheap":
                    # Elektrik ucuzsa başlat
                    actions.append(1)
                else:
                    # Pahalıysa beklet
                    actions.append(0)

            # Sürekli cihaz
            elif device.device_type == DEVICE_TYPE_CONTINUOUS:
                if device.comfort_sensitive:
                    # Klima: sıcaklık konfor dışına çıktıysa aç
                    if indoor_temp < temp_min or indoor_temp > temp_max:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    # Aydınlatma: gece ve kullanıcı uyanıksa aç
                    if is_night and user_awake:
                        actions.append(1)
                    else:
                        actions.append(0)

            else:
                actions.append(0)

        # Slot sayısı kontrolü
        assert len(actions) == len(slot_manager.slots)

        return actions
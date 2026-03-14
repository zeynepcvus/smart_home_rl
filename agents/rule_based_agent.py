from environment.pricing import get_price_category

class RuleBasedAgent:
    """
    Basit kural tabanlı kontrolcü.
    RL ajanıyla karşılaştırmak için kullanılacak.

    Kurallar:
    - Elektrik ucuzsa cihazları çalıştır
    - Sıcaklık konfor dışına çıktıysa klimayı aç
    - Işığı her zaman yarıda tut
    """

    def select_action(self, observation):       # observation 8 sayılık listeydi saat, iç sıcaklık, dış sıcaklık vs...
        hour = int(observation[0])  # index 0 → saat
        indoor_temp = float(observation[2])  # index 2 → iç sıcaklık
        washer_pend = bool(observation[7])  # index 7 → çamaşır bekliyor mu?

        price_cat = get_price_category(hour)    # Saate göre "cheap", "normal", "expensive" döndürüyor (daha önce yazdığım fonk)

        # 1. KLİMA: sıcaklık konfor dışına çıktıysa aç
        hvac_action = 1 if (indoor_temp < 20.0 or indoor_temp > 24.0) else 0

        # 2. ÇAMAŞIR: sadece ucuz saatlerde başlat
        washer_action = 1 if (washer_pend and price_cat == "cheap") else 0

        # 3. IŞIK: her zaman yarıda
        light_action = 1

        return [hvac_action, washer_action, light_action]

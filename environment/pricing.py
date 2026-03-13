import numpy as np

HOURLY_PRICES = np.array([
    0.8, 0.8, 0.8, 0.8,  # 00-03: gece (ucuz)
    0.8, 0.8, 1.2, 1.2,  # 04-07: sabah
    2.5, 2.5, 2.5, 2.5,  # 08-11: sabah pik (pahalı)
    2.5, 2.5, 1.5, 1.5,  # 12-15: öğleden sonra
    2.5, 2.5, 2.5, 2.5,  # 16-19: akşam pik (pahalı)
    1.5, 1.2, 1.0, 0.8,  # 20-23: gece geçişi
])

# Verilen saatteki elektrik fiyatını döndür
def get_price(hour: int) -> float:
    base_price = HOURLY_PRICES[hour % 24]   # Saati her zaman 0-23 arasında tutar. (Yani 25 versem bile 1 olarak algılar.)
    price = base_price * (1 + np.random.uniform(-0.05, 0.05))   # Fiyata küçük bir rastgelelik ekler. Gerçek hayattaki fiyat dalgalanmalarını taklit etmek için.
    return round(float(price), 3)

# Fiyat kategorisi döndür: cheap / normal / expensive
def get_price_category(hour: int) -> str:
    price = HOURLY_PRICES[hour % 24]
    if price <=1.0:
        return "cheap"
    elif price <= 1.8:
        return "normal"
    else:
        return "expensive"
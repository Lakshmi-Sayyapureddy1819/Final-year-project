import requests

def get_live_sst(lat, lon):
    api = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&daily=sea_surface_temperature_mean&timezone=auto"
    r = requests.get(api).json()
    try:
        return r["daily"]["sea_surface_temperature_mean"][0]
    except:
        return None

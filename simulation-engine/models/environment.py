import numpy as np

def diurnal_temperature(t_hours: float, t_mean=22.0, amplitude=5.0, phi=-1.2) -> float:
    """
    T(t) = T_mean + A * sin(2π t / 24 + φ)
    phi=-1.2 places peak temperature around 14:00 (mid-afternoon)
    """
    return t_mean + amplitude * np.sin((2 * np.pi * t_hours / 24) + phi)


def vpd(temperature_c: float, rh_percent: float) -> float:
    """
    VP_sat = 0.61078 * exp(17.27 * T / (T + 237.3))
    VPD = VP_sat - VP_air,  VP_air = VP_sat * RH/100
    Returns kPa
    """
    vp_sat = 0.61078 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    vp_air = vp_sat * (rh_percent / 100.0)
    return round(vp_sat - vp_air, 4)


def vpd_status(vpd_kpa: float) -> str:
    if vpd_kpa < 0.5:
        return "too_low"
    elif vpd_kpa > 2.0:
        return "too_high"
    return "optimal"
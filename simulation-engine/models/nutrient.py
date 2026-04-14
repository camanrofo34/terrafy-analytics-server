def michaelis_menten(c: float, j_max: float, k_m: float) -> float:
    """
    J = J_max * C / (K_m + C)
    c     : current nutrient concentration (mmol/L)
    j_max : max absorption rate for current growth stage (mmol/h)
    k_m   : Michaelis constant (mmol/L)
    Returns absorption rate in mmol/h
    """
    if c <= 0:
        return 0.0
    return j_max * c / (k_m + c)


# Lettuce reference constants per nutrient
NUTRIENT_PARAMS = {
    "N": {"j_max": 2.8,  "k_m": 0.08},
    "P": {"j_max": 0.32, "k_m": 0.01},
    "K": {"j_max": 3.10, "k_m": 0.10},
}


def update_concentration(c: float, absorption: float, dt_hours: float) -> float:
    """Reduce tank concentration by absorbed amount over dt."""
    delta = absorption * dt_hours
    return max(0.0, round(c - delta, 4))
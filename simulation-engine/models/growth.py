import numpy as np

def plant_root_length(t_hours: float, r_max=28.0, k1=6.0, k1_rate=0.035) -> float:
    """
    PRL = R_max / (1 + e^(K1 - k1*t))
    r_max   : max root length cm (lettuce ~28cm)
    k1      : inflection point shift
    k1_rate : growth rate constant
    t       : hours since transplant
    """
    return round(r_max / (1 + np.exp(k1 - k1_rate * t_hours)), 3)


def growth_stage(t_hours: float) -> str:
    """Categorical label for the growth phase."""
    if t_hours < 72:
        return "seedling"
    elif t_hours < 336:
        return "vegetative"
    elif t_hours < 600:
        return "mature"
    return "harvest_ready"
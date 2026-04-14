import time, os
from datetime import datetime, timezone
from dotenv import load_dotenv

from models.environment import diurnal_temperature, vpd, vpd_status
from models.nutrient import michaelis_menten, update_concentration, NUTRIENT_PARAMS
from models.growth import plant_root_length, growth_stage
from models.sensor import OUProcess
from publisher import publish_reading

load_dotenv()

INTERVAL  = int(os.getenv("SIM_INTERVAL_SECONDS", 5))
DT_HOURS  = INTERVAL / 3600.0

# ---------------------------------------------------------------------------
# System configurations — one entry per agronomic system
# ---------------------------------------------------------------------------
SYSTEM_CONFIGS = [
    {
        "system_id":  "1",
        "name":       "Sistema de Lechugas NFT A",
        "rh":         70.0,
        "temp_mean":  22.0,
        "temp_amp":   5.0,
        "ph_mu":      6.0,  "ec_mu": 1.8,  "do_mu": 7.5,
        "N0": 8.0, "P0": 1.2, "K0": 6.0,
    },
    {
        "system_id":  "2",
        "name":       "Sistema DWC B",
        "rh":         65.0,
        "temp_mean":  24.0,
        "temp_amp":   4.0,
        "ph_mu":      6.2,  "ec_mu": 2.0,  "do_mu": 8.0,
        "N0": 9.0, "P0": 1.5, "K0": 7.0,
    },
    {
        "system_id":  "3",
        "name":       "Flujo de fresas C",
        "rh":         75.0,
        "temp_mean":  20.0,
        "temp_amp":   3.0,
        "ph_mu":      5.8,  "ec_mu": 1.5,  "do_mu": 7.0,
        "N0": 7.0, "P0": 1.0, "K0": 5.5,
    },
]

# ---------------------------------------------------------------------------
# Per-system mutable state (concentrations, elapsed time, sensor drifters)
# ---------------------------------------------------------------------------
def _init_system(cfg: dict) -> dict:
    return {
        "t_hours": 0.0,
        "concentrations": {"N": cfg["N0"], "P": cfg["P0"], "K": cfg["K0"]},
        "sensors": {
            "ph": OUProcess(mu=cfg["ph_mu"], theta=0.05, sigma=0.015),
            "ec": OUProcess(mu=cfg["ec_mu"], theta=0.08, sigma=0.030),
            "do": OUProcess(mu=cfg["do_mu"], theta=0.10, sigma=0.050),
        },
    }

states = [_init_system(cfg) for cfg in SYSTEM_CONFIGS]

print(f"Simulation engine started — {len(SYSTEM_CONFIGS)} systems, tick every {INTERVAL}s.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    for cfg, state in zip(SYSTEM_CONFIGS, states):
        t = state["t_hours"]

        # Environment (each system has its own microclimate)
        temp    = diurnal_temperature(t % 24, t_mean=cfg["temp_mean"], amplitude=cfg["temp_amp"])
        vpd_kpa = vpd(temp, cfg["rh"])

        # Plant growth
        prl   = plant_root_length(t)
        stage = growth_stage(t)

        # Nutrient uptake
        area_factor = min(prl / 28.0, 1.0)
        absorption  = {}
        for ion, params in NUTRIENT_PARAMS.items():
            j = michaelis_menten(
                c     = state["concentrations"][ion],
                j_max = params["j_max"] * area_factor,
                k_m   = params["k_m"],
            )
            absorption[ion] = round(j, 5)
            state["concentrations"][ion] = update_concentration(
                state["concentrations"][ion], j, DT_HOURS
            )

        # Sensor readings with OU drift
        ph_reading = state["sensors"]["ph"].step()
        ec_reading = state["sensors"]["ec"].step()
        do_reading = state["sensors"]["do"].step()

        reading = {
            "system_id":    cfg["system_id"],
            "system_name":  cfg["name"],
            "t_hours":      round(t, 4),
            "growth_stage": stage,
            "environment": {
                "temperature_c": round(temp, 3),
                "rh_percent":    cfg["rh"],
                "vpd_kpa":       vpd_kpa,
                "vpd_status":    vpd_status(vpd_kpa),
            },
            "plant": {
                "root_length_cm": prl,
            },
            "sensors": {
                "ph":           ph_reading,
                "ec_ms_cm":     ec_reading,
                "dissolved_o2": do_reading,
            },
            "concentrations":  dict(state["concentrations"]),
            "absorption_rates": absorption,
        }

        publish_reading(reading)
        print(
            f"[{cfg['system_id']} | t={t:.2f}h | {stage}] "
            f"T={temp:.1f}°C  VPD={vpd_kpa:.3f}kPa  "
            f"pH={ph_reading}  EC={ec_reading}"
        )

        state["t_hours"] += DT_HOURS

    time.sleep(INTERVAL)

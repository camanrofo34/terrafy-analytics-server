"""
Agronomic variable and system registry.
Single source of truth shared across routes.
"""

VARIABLES = {
    "1": {
        "variableId": "1",
        "name": "Temperatura",
        "measurementUnit": "°C",
        "field": "environment.temperature_c",
    },
    "2": {
        "variableId": "2",
        "name": "Humedad Relativa",
        "measurementUnit": "%",
        "field": "environment.rh_percent",
    },
    "3": {
        "variableId": "3",
        "name": "Déficit de la presión de vapor",
        "measurementUnit": "kPa",
        "field": "environment.vpd_kpa",
    },
    "4": {
        "variableId": "4",
        "name": "pH",
        "measurementUnit": "pH",
        "field": "sensors.ph",
    },
    "5": {
        "variableId": "5",
        "name": "Conductividad Eléctrica",
        "measurementUnit": "mS/cm",
        "field": "sensors.ec_ms_cm",
    },
    "6": {
        "variableId": "6",
        "name": "Oxígeno Disuelto",
        "measurementUnit": "mg/L",
        "field": "sensors.dissolved_o2",
    },
    "7": {
        "variableId": "7",
        "name": "Nitrógeno",
        "measurementUnit": "mmol/L",
        "field": "concentrations.N",
    },
    "8": {
        "variableId": "8",
        "name": "Fósforo",
        "measurementUnit": "mmol/L",
        "field": "concentrations.P",
    },
    "9": {
        "variableId": "9",
        "name": "Potasio",
        "measurementUnit": "mmol/L",
        "field": "concentrations.K",
    },
    "10": {
        "variableId": "10",
        "name": "Altura del cultivo",
        "measurementUnit": "cm",
        "field": "plant.root_length_cm",
    },
}

SYSTEMS = {
    "1": {
        "systemId": "1",
        "name": "Sistema de Lechugas NFT A",
        "description": "Nutrient Film Technique system for lettuce cultivation",
    },
    "2": {
        "systemId": "2",
        "name": "Sistema DWC B",
        "description": "Deep Water Culture system for basil production",
    },
    "3": {
        "systemId": "3",
        "name": "Flujo de fresas C",
        "description": "Ebb and Flow system for strawberry cultivation",
    },
}

# Alert thresholds — keys match VARIABLES numeric IDs
ALERT_RULES = {
    "3": {   # VPD
        "field": "environment.vpd_kpa",
        "min": 0.5,
        "max": 2.0,
        "unit": "kPa",
    },
    "4": {   # pH
        "field": "sensors.ph",
        "min": 5.5,
        "max": 6.5,
        "unit": "pH",
    },
    "5": {   # EC
        "field": "sensors.ec_ms_cm",
        "min": 1.2,
        "max": 2.5,
        "unit": "mS/cm",
    },
    "7": {   # Nitrogen
        "field": "concentrations.N",
        "min": 1.0,
        "max": None,
        "unit": "mmol/L",
    },
}

GROUPING_UNITS = {
    "minutes": "minute",
    "hours": "hour",
    "days": "day",
    "weeks": "week",
}


def get_nested(doc: dict, field: str):
    """Traverse a dot-notation field path into a nested dict."""
    parts = field.split(".")
    val = doc
    for p in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(p)
    return val

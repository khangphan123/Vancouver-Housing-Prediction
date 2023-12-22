from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import sys
import re
import pandas as pd
import numpy as np


geolocator = Nominatim(
    user_agent="cmpt353_project", domain="nominatim.darksunlight.xyz"
)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1/20)

def property_to_coords(property):
    location = geocode(
        f"{int(property['CIVIC_NUMBER'])} {property['STREET_NAME']}, Vancouver",
        country_codes="ca",
    )
    if (
        location == None
        and isinstance(property["PROPERTY_POSTAL_CODE"], str)
        and property["PROPERTY_POSTAL_CODE"] != ""
    ):
        print(f"{property['PID']} using postal code")
        location = geocode(f"{property['PROPERTY_POSTAL_CODE']}", country_codes="ca")
        if location == None:
            location = geocode(
                f"{property['PROPERTY_POSTAL_CODE'][:3]}", country_codes="ca"
            )

    return pd.Series(
        {
            "PID": property["PID"],
            "CIVIC_NUMBER": property["CIVIC_NUMBER"],
            "STREET_NAME": property["STREET_NAME"],
            "lat": location.latitude if location != None else np.nan,
            "lon": location.longitude if location != None else np.nan,
        }
    )


def normalize_street_name(name):
    name = re.sub(r"PLACE$", "PL", name)
    name = re.sub(r"SQUARE$", "SQ", name)
    name = re.sub(r"AV$", "AVE", name)
    name = re.sub(r"^([WE]) (.*)$", r"\2 \1", name)
    return name


def main(pid_addresses_path, property_coords_path):
    pid_addresses = pd.read_csv(pid_addresses_path)
    property_coords = pd.read_csv(
        property_coords_path, sep=";", dtype={"STD_STREET": "string"}
    )
    property_coords = property_coords[property_coords["STD_STREET"].notna()]
    pid_addresses["CIVIC_NUMBER"] = pid_addresses["TO_CIVIC_NUMBER"]
    property_coords["STREET_NAME"] = property_coords["STD_STREET"].apply(
        normalize_street_name
    )
    property_coords[["lat", "lon"]] = property_coords["geo_point_2d"].str.split(
        ", ", expand=True
    )
    merged = pid_addresses.merge(property_coords, on=["CIVIC_NUMBER", "STREET_NAME"])
    merged = merged[["PID", "CIVIC_NUMBER", "STREET_NAME", "lat", "lon"]]

    missing = pid_addresses[~pid_addresses["PID"].isin(merged["PID"])]
    missing = missing.apply(property_to_coords, axis=1)
    merged = pd.concat([merged, missing])
    merged.to_csv("pid_coords.csv", index=False)


if __name__ == "__main__":
    pid_addresses_path = sys.argv[1]
    property_coords_path = sys.argv[2]
    main(pid_addresses_path, property_coords_path)

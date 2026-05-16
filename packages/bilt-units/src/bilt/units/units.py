"""Common units defined with the (mostly SI) units system."""

__all__ = (
    "DIMENSIONLESS",
    "EUR",
    "GBP",
    "JPY",
    "USD",
    "absolute_celsius",
    "absolute_fahrenheit",
    "absolute_kelvin",
    "absolute_year",
    "acre",
    "ampere",
    "atmosphere",
    "bar",
    "british_thermal_unit",
    "candela",
    "celsius",
    "centimeter",
    "centipoise",
    "coulomb",
    "day",
    "degree",
    "exajoule",
    "exawatt",
    "fahrenheit",
    "foot",
    "fraction",
    "gigagram",
    "gigajoule",
    "gigawatt",
    "gigawatthour",
    "gram",
    "hectare",
    "hour",
    "inch",
    "joule",
    "kelvin",
    "kilogram",
    "kilojoule",
    "kilometer",
    "kilomole",
    "kilonewton",
    "kilopascal",
    "kilotonne",
    "kilowatt",
    "kilowatthour",
    "liter",
    "megagram",
    "megajoule",
    "meganewton",
    "megapascal",
    "megatonne",
    "megawatt",
    "megawatthour",
    "meter",
    "mile",
    "milligram",
    "millimeter",
    "millinewton",
    "million_british_thermal_unit",
    "millisecond",
    "millivolt",
    "minute",
    "mole",
    "newton",
    "ohm",
    "pascal",
    "percent",
    "petajoule",
    "petawatt",
    "poise",
    "poiseuille",
    "radian",
    "second",
    "siemen",
    "teragram",
    "terajoule",
    "terawatt",
    "therm",
    "tonne",
    "torr",
    "volt",
    "watt",
    "watthour",
    "yard",
    "year",
)

import math

from bilt.units._currencies import EUR, GBP, JPY, USD

from ._prefix import Prefix
from ._units import (
    DIMENSIONLESS,
    Unit,
    ampere,
    candela,
    kelvin,
    kilogram,
    meter,
    mole,
    second,
)

# Dimensionless
radian = Unit.new_named("radian", relation=DIMENSIONLESS)
degree = Unit.new_named("degree", relation=(math.pi / 180.0) * radian)
fraction = Unit.new_named("fraction", relation=DIMENSIONLESS)
percent = Unit.new_named("percent", relation=0.01 * fraction)

# Mass
milligram = Unit.new_named("milligram", relation=kilogram, prefix=Prefix.MILLI)
gram = Unit.new_named("gram", relation=kilogram)
megagram = Unit.new_named("megagram", relation=kilogram, prefix=Prefix.MEGA)
gigagram = Unit.new_named("gigagram", relation=kilogram, prefix=Prefix.GIGA)
teragram = Unit.new_named("teragram", relation=kilogram, prefix=Prefix.TERA)
tonne = Unit.new_named("tonne", relation=1_000.0 * kilogram)
kilotonne = Unit.new_named("kilotonne", relation=tonne, prefix=Prefix.KILO)
megatonne = Unit.new_named("megatonne", relation=tonne, prefix=Prefix.MEGA)

# Length
millimeter = Unit.new_named("millimeter", relation=meter, prefix=Prefix.MILLI)
centimeter = Unit.new_named("centimeter", relation=meter, prefix=Prefix.CENTI)
kilometer = Unit.new_named("kilometer", relation=meter, prefix=Prefix.KILO)
foot = Unit.new_named("foot", relation=0.3048 * meter)
inch = Unit.new_named("inch", relation=(1.0 / 12.0) * foot)
yard = Unit.new_named("yard", relation=3.0 * foot)
mile = Unit.new_named("mile", relation=1_609.344 * meter)

# Time
millisecond = Unit.new_named("millisecond", relation=second, prefix=Prefix.MILLI)
minute = Unit.new_named("minute", relation=60.0 * second)
hour = Unit.new_named("hour", relation=60.0 * minute)
day = Unit.new_named("day", relation=24.0 * hour)
year = Unit.new_named("year", relation=365.25 * day)
absolute_year = Unit.new_absolute("|year|", scale=year)

# Temperature
celsius = Unit.new_named("celsius", kelvin)
fahrenheit = Unit.new_named("fahrenheit", (5.0 / 9.0) * celsius)
absolute_kelvin = Unit.new_absolute("|kelvin|", scale=kelvin)
absolute_celsius = Unit.new_absolute("|celsius|", scale=celsius, zero=273.15 * abs(kelvin))
absolute_fahrenheit = Unit.new_absolute("|fahrenheit|", scale=fahrenheit, zero=-160.0 / 9.0 * abs(celsius))

# Amount
kilomole = Unit.new_named("kilomole", relation=mole, prefix=Prefix.KILO)

# Area
acre = Unit.new_named("acre", relation=4_840.0 * yard**2)
hectare = Unit.new_named("hectare", relation=10_000.0 * meter**2)

# Volume
liter = Unit.new_named("liter", relation=0.001 * meter**3)

# Force
newton = Unit.new_named("newton", relation=kilogram * meter / second**2)
millinewton = Unit.new_named("millinewton", relation=newton, prefix=Prefix.MILLI)
kilonewton = Unit.new_named("kilonewton", relation=newton, prefix=Prefix.KILO)
meganewton = Unit.new_named("meganewton", relation=newton, prefix=Prefix.MEGA)

# Pressure
pascal = Unit.new_named("pascal", relation=newton / meter**2)
kilopascal = Unit.new_named("kilopascal", relation=pascal, prefix=Prefix.KILO)
megapascal = Unit.new_named("megapascal", relation=pascal, prefix=Prefix.MEGA)
bar = Unit.new_named("bar", relation=100_000.0 * pascal)
atmosphere = Unit.new_named("atmosphere", relation=101_325.0 * pascal)
torr = Unit.new_named("torr", relation=1.0 / 760.0 * atmosphere)

# Power
watt = Unit.new_named("watt", relation=kilogram * meter**2 / second**3)
kilowatt = Unit.new_named("kilowatt", relation=watt, prefix=Prefix.KILO)
megawatt = Unit.new_named("megawatt", relation=watt, prefix=Prefix.MEGA)
gigawatt = Unit.new_named("gigawatt", relation=watt, prefix=Prefix.GIGA)
terawatt = Unit.new_named("terawatt", relation=watt, prefix=Prefix.TERA)
petawatt = Unit.new_named("petawatt", relation=watt, prefix=Prefix.PETA)
exawatt = Unit.new_named("exawatt", relation=watt, prefix=Prefix.EXA)

# Energy
joule = Unit.new_named("joule", relation=kilogram * meter**2 / second**2)
kilojoule = Unit.new_named("kilojoule", relation=joule, prefix=Prefix.KILO)
megajoule = Unit.new_named("megajoule", relation=joule, prefix=Prefix.MEGA)
gigajoule = Unit.new_named("gigajoule", relation=joule, prefix=Prefix.GIGA)
terajoule = Unit.new_named("terajoule", relation=joule, prefix=Prefix.TERA)
petajoule = Unit.new_named("petajoule", relation=joule, prefix=Prefix.PETA)
exajoule = Unit.new_named("exajoule", relation=joule, prefix=Prefix.EXA)
watthour = Unit.new_named("watthour", relation=watt * hour)
kilowatthour = Unit.new_named("kilowatthour", relation=watthour, prefix=Prefix.KILO)
megawatthour = Unit.new_named("megawatthour", relation=watthour, prefix=Prefix.MEGA)
gigawatthour = Unit.new_named("gigawatthour", relation=watthour, prefix=Prefix.GIGA)
british_thermal_unit = Unit.new_named("british_thermal_unit", relation=1055.0 * joule)
therm = Unit.new_named("therm", relation=100_000.0 * british_thermal_unit)
million_british_thermal_unit = Unit.new_named(
    "million_british_thermal_unit",
    relation=1_000_000.0 * british_thermal_unit,
)

# Voltage
volt = Unit.new_named("volt", relation=joule / (ampere * second))
millivolt = Unit.new_named("millivolt", relation=volt, prefix=Prefix.MILLI)

# Electric charge
coulomb = Unit.new_named("coulomb", relation=ampere * second)

# Electric resistance
ohm = Unit.new_named("ohm", relation=(kilogram * meter**2) / (second**3 * ampere**2))

# Electric conductance
siemen = Unit.new_named("siemen", relation=(second**3 * ampere**2) / (kilogram * meter**2))

# Viscosity
poiseuille = Unit.new_named("poiseuille", relation=pascal * second)
poise = Unit.new_named("poise", relation=0.1 * poiseuille)
centipoise = Unit.new_named("centipoise", relation=poiseuille, prefix=Prefix.CENTI)

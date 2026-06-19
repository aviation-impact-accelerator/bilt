import enum


@enum.unique
class Prefix(enum.IntEnum):
    ATTO = -18
    FEMTO = -15
    PICO = -12
    NANO = -9
    MICRO = -6
    MILLI = -3
    CENTI = -2
    DECI = -1
    DECA = 1
    HECTO = 2
    KILO = 3
    MEGA = 6
    GIGA = 9
    TERA = 12
    PETA = 15
    EXA = 18


PREFIX_NAMES = {
    Prefix.ATTO: "atto",
    Prefix.FEMTO: "femto",
    Prefix.PICO: "pico",
    Prefix.NANO: "nano",
    Prefix.MICRO: "micro",
    Prefix.MILLI: "milli",
    Prefix.CENTI: "centi",
    Prefix.DECI: "deci",
    Prefix.DECA: "deca",
    Prefix.HECTO: "hecto",
    Prefix.KILO: "kilo",
    Prefix.MEGA: "mega",
    Prefix.GIGA: "giga",
    Prefix.TERA: "tera",
    Prefix.PETA: "peta",
    Prefix.EXA: "exa",
}

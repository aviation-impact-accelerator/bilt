__all__ = ["EUR", "GBP", "JPY", "USD"]

import dataclasses
import typing

from ._units import _USD2025, Unit


@typing.final
@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class AnnualCurrencyUnit:
    """Mapping of years as `int` literals to the corresponding currency unit.

    Example:
        To get the unit for USD in 2024, index into the `USD` variable using the `int` literal
        corresponding to the year, so `2024`.

        >>> from bilt.units import USD

        >>> str(USD[2024])
        'USD2024'

        As these currency units are the same as all other units in the units system, we can convert
        between them using the `convert_to` method. The conversion will correctly apply any exchange
        rates and inflation calculations.

        Inflation is the decline in the purchasing power of a currency, so purchasing something in
        the past would have required fewer units of that currency. This can alternatively be thought
        of in terms of the price of something increasing into the future. For example, between 2023
        and 2024, USD inflation was 2.9% so $1.00 in 2024 currency should convert to approximately
        $0.97 in 2023 currency. Going back further, that same $1.00 in 2024 currency would have been
        worth approximately $0.55 in 2000 currency.

        >>> str((1.0 * USD[2024]).convert_to(USD[2023]))
        '0.9718172983479105 USD2023'
        >>> str((1.0 * USD[2024]).convert_to(USD[2000]))
        '0.5502044598903999 USD2000'

        Likewise, we can convert between different currencies and the units system will do the
        handling of the exchange rate appropriate to the year in question. In 2024, $1.00 was worth,
        on average, approximately £0.78.

        >>> from bilt.units import GBP

        >>> str((1.0 * USD[2024]).convert_to(GBP[2024]))
        '0.7825 GBP2024'
        >>> str((1.0 * GBP[2024]).convert_to(USD[2024]))
        '1.2779552715654952 USD2024'

    """

    name: str = dataclasses.field(kw_only=False)

    _2025: Unit
    _2024: Unit
    _2023: Unit
    _2022: Unit
    _2021: Unit
    _2020: Unit
    _2019: Unit
    _2018: Unit
    _2017: Unit
    _2016: Unit
    _2015: Unit
    _2014: Unit
    _2013: Unit
    _2012: Unit
    _2011: Unit
    _2010: Unit
    _2009: Unit
    _2008: Unit
    _2007: Unit
    _2006: Unit
    _2005: Unit
    _2004: Unit
    _2003: Unit
    _2002: Unit
    _2001: Unit
    _2000: Unit

    def __getitem__(self, key: int) -> Unit:
        return typing.cast("Unit", getattr(self, f"_{key!s}"))

    def __str__(self) -> str:
        return self.name


_USD2024 = Unit.new_named("USD2024", relation=1.029 * _USD2025)
_USD2023 = Unit.new_named("USD2023", relation=1.029 * _USD2024)
_USD2022 = Unit.new_named("USD2022", relation=1.041 * _USD2023)
_USD2021 = Unit.new_named("USD2021", relation=1.080 * _USD2022)
_USD2020 = Unit.new_named("USD2020", relation=1.047 * _USD2021)
_USD2019 = Unit.new_named("USD2019", relation=1.012 * _USD2020)
_USD2018 = Unit.new_named("USD2018", relation=1.018 * _USD2019)
_USD2017 = Unit.new_named("USD2017", relation=1.024 * _USD2018)
_USD2016 = Unit.new_named("USD2016", relation=1.021 * _USD2017)
_USD2015 = Unit.new_named("USD2015", relation=1.013 * _USD2016)
_USD2014 = Unit.new_named("USD2014", relation=1.001 * _USD2015)
_USD2013 = Unit.new_named("USD2013", relation=1.016 * _USD2014)
_USD2012 = Unit.new_named("USD2012", relation=1.015 * _USD2013)
_USD2011 = Unit.new_named("USD2011", relation=1.021 * _USD2012)
_USD2010 = Unit.new_named("USD2010", relation=1.032 * _USD2011)
_USD2009 = Unit.new_named("USD2009", relation=1.016 * _USD2010)
_USD2008 = Unit.new_named("USD2008", relation=0.996 * _USD2009)
_USD2007 = Unit.new_named("USD2007", relation=1.038 * _USD2008)
_USD2006 = Unit.new_named("USD2006", relation=1.028 * _USD2007)
_USD2005 = Unit.new_named("USD2005", relation=1.032 * _USD2006)
_USD2004 = Unit.new_named("USD2004", relation=1.034 * _USD2005)
_USD2003 = Unit.new_named("USD2003", relation=1.027 * _USD2004)
_USD2002 = Unit.new_named("USD2002", relation=1.023 * _USD2003)
_USD2001 = Unit.new_named("USD2001", relation=1.016 * _USD2002)
_USD2000 = Unit.new_named("USD2000", relation=1.028 * _USD2001)

_GBP2025 = Unit.new_named("GBP2025", relation=(1.0 / 0.7435) * _USD2025)
_GBP2024 = Unit.new_named("GBP2024", relation=(1.0 / 0.7825) * _USD2024)
_GBP2023 = Unit.new_named("GBP2023", relation=(1.0 / 0.8046) * _USD2023)
_GBP2022 = Unit.new_named("GBP2022", relation=(1.0 / 0.8120) * _USD2022)
_GBP2021 = Unit.new_named("GBP2021", relation=(1.0 / 0.7271) * _USD2021)
_GBP2020 = Unit.new_named("GBP2020", relation=(1.0 / 0.7797) * _USD2020)
_GBP2019 = Unit.new_named("GBP2019", relation=(1.0 / 0.7839) * _USD2019)
_GBP2018 = Unit.new_named("GBP2018", relation=(1.0 / 0.7501) * _USD2018)
_GBP2017 = Unit.new_named("GBP2017", relation=(1.0 / 0.7765) * _USD2017)
_GBP2016 = Unit.new_named("GBP2016", relation=(1.0 / 0.7414) * _USD2016)
_GBP2015 = Unit.new_named("GBP2015", relation=(1.0 / 0.6544) * _USD2015)
_GBP2014 = Unit.new_named("GBP2014", relation=(1.0 / 0.6074) * _USD2014)
_GBP2013 = Unit.new_named("GBP2013", relation=(1.0 / 0.6397) * _USD2013)
_GBP2012 = Unit.new_named("GBP2012", relation=(1.0 / 0.6310) * _USD2012)
_GBP2011 = Unit.new_named("GBP2011", relation=(1.0 / 0.6240) * _USD2011)
_GBP2010 = Unit.new_named("GBP2010", relation=(1.0 / 0.6475) * _USD2010)
_GBP2009 = Unit.new_named("GBP2009", relation=(1.0 / 0.6408) * _USD2009)
_GBP2008 = Unit.new_named("GBP2008", relation=(1.0 / 0.5454) * _USD2008)
_GBP2007 = Unit.new_named("GBP2007", relation=(1.0 / 0.4996) * _USD2007)
_GBP2006 = Unit.new_named("GBP2006", relation=(1.0 / 0.5434) * _USD2006)
_GBP2005 = Unit.new_named("GBP2005", relation=(1.0 / 0.5504) * _USD2005)
_GBP2004 = Unit.new_named("GBP2004", relation=(1.0 / 0.5460) * _USD2004)
_GBP2003 = Unit.new_named("GBP2003", relation=(1.0 / 0.6121) * _USD2003)
_GBP2002 = Unit.new_named("GBP2002", relation=(1.0 / 0.6663) * _USD2002)
_GBP2001 = Unit.new_named("GBP2001", relation=(1.0 / 0.6947) * _USD2001)
_GBP2000 = Unit.new_named("GBP2000", relation=(1.0 / 0.6613) * _USD2000)

_EUR2025 = Unit.new_named("EUR2025", relation=(1.0 / 0.8588) * _USD2025)
_EUR2024 = Unit.new_named("EUR2024", relation=(1.0 / 0.9245) * _USD2024)
_EUR2023 = Unit.new_named("EUR2023", relation=(1.0 / 0.9251) * _USD2023)
_EUR2022 = Unit.new_named("EUR2022", relation=(1.0 / 0.9517) * _USD2022)
_EUR2021 = Unit.new_named("EUR2021", relation=(1.0 / 0.8459) * _USD2021)
_EUR2020 = Unit.new_named("EUR2020", relation=(1.0 / 0.8771) * _USD2020)
_EUR2019 = Unit.new_named("EUR2019", relation=(1.0 / 0.8935) * _USD2019)
_EUR2018 = Unit.new_named("EUR2018", relation=(1.0 / 0.8477) * _USD2018)
_EUR2017 = Unit.new_named("EUR2017", relation=(1.0 / 0.8866) * _USD2017)
_EUR2016 = Unit.new_named("EUR2016", relation=(1.0 / 0.9041) * _USD2016)
_EUR2015 = Unit.new_named("EUR2015", relation=(1.0 / 0.9018) * _USD2015)
_EUR2014 = Unit.new_named("EUR2014", relation=(1.0 / 0.7539) * _USD2014)
_EUR2013 = Unit.new_named("EUR2013", relation=(1.0 / 0.7531) * _USD2013)
_EUR2012 = Unit.new_named("EUR2012", relation=(1.0 / 0.7785) * _USD2012)
_EUR2011 = Unit.new_named("EUR2011", relation=(1.0 / 0.7194) * _USD2011)
_EUR2010 = Unit.new_named("EUR2010", relation=(1.0 / 0.7553) * _USD2010)
_EUR2009 = Unit.new_named("EUR2009", relation=(1.0 / 0.7192) * _USD2009)
_EUR2008 = Unit.new_named("EUR2008", relation=(1.0 / 0.6838) * _USD2008)
_EUR2007 = Unit.new_named("EUR2007", relation=(1.0 / 0.7306) * _USD2007)
_EUR2006 = Unit.new_named("EUR2006", relation=(1.0 / 0.7969) * _USD2006)
_EUR2005 = Unit.new_named("EUR2005", relation=(1.0 / 0.8053) * _USD2005)
_EUR2004 = Unit.new_named("EUR2004", relation=(1.0 / 0.8049) * _USD2004)
_EUR2003 = Unit.new_named("EUR2003", relation=(1.0 / 0.8851) * _USD2003)
_EUR2002 = Unit.new_named("EUR2002", relation=(1.0 / 1.0606) * _USD2002)
_EUR2001 = Unit.new_named("EUR2001", relation=(1.0 / 1.1177) * _USD2001)
_EUR2000 = Unit.new_named("EUR2000", relation=(1.0 / 1.0863) * _USD2000)

_JPY2025 = Unit.new_named("JPY2025", relation=(1.0 / 147.5010) * _USD2025)
_JPY2024 = Unit.new_named("JPY2024", relation=(1.0 / 151.4655) * _USD2024)
_JPY2023 = Unit.new_named("JPY2023", relation=(1.0 / 140.6448) * _USD2023)
_JPY2022 = Unit.new_named("JPY2022", relation=(1.0 / 131.5526) * _USD2022)
_JPY2021 = Unit.new_named("JPY2021", relation=(1.0 / 109.8402) * _USD2021)
_JPY2020 = Unit.new_named("JPY2020", relation=(1.0 / 106.7765) * _USD2020)
_JPY2019 = Unit.new_named("JPY2019", relation=(1.0 / 109.0025) * _USD2019)
_JPY2018 = Unit.new_named("JPY2018", relation=(1.0 / 110.4741) * _USD2018)
_JPY2017 = Unit.new_named("JPY2017", relation=(1.0 / 112.1617) * _USD2017)
_JPY2016 = Unit.new_named("JPY2016", relation=(1.0 / 108.6315) * _USD2016)
_JPY2015 = Unit.new_named("JPY2015", relation=(1.0 / 121.0790) * _USD2015)
_JPY2014 = Unit.new_named("JPY2014", relation=(1.0 / 105.8407) * _USD2014)
_JPY2013 = Unit.new_named("JPY2013", relation=(1.0 / 97.5888) * _USD2013)
_JPY2012 = Unit.new_named("JPY2012", relation=(1.0 / 79.7736) * _USD2012)
_JPY2011 = Unit.new_named("JPY2011", relation=(1.0 / 79.6747) * _USD2011)
_JPY2010 = Unit.new_named("JPY2010", relation=(1.0 / 87.6853) * _USD2010)
_JPY2009 = Unit.new_named("JPY2009", relation=(1.0 / 93.5641) * _USD2009)
_JPY2008 = Unit.new_named("JPY2008", relation=(1.0 / 103.4306) * _USD2008)
_JPY2007 = Unit.new_named("JPY2007", relation=(1.0 / 117.7615) * _USD2007)
_JPY2006 = Unit.new_named("JPY2006", relation=(1.0 / 116.3016) * _USD2006)
_JPY2005 = Unit.new_named("JPY2005", relation=(1.0 / 110.1987) * _USD2005)
_JPY2004 = Unit.new_named("JPY2004", relation=(1.0 / 108.1613) * _USD2004)
_JPY2003 = Unit.new_named("JPY2003", relation=(1.0 / 115.8845) * _USD2003)
_JPY2002 = Unit.new_named("JPY2002", relation=(1.0 / 125.1515) * _USD2002)
_JPY2001 = Unit.new_named("JPY2001", relation=(1.0 / 121.4508) * _USD2001)
_JPY2000 = Unit.new_named("JPY2000", relation=(1.0 / 107.7925) * _USD2000)

USD = AnnualCurrencyUnit(
    name="USD",
    _2000=_USD2000,
    _2001=_USD2001,
    _2002=_USD2002,
    _2003=_USD2003,
    _2004=_USD2004,
    _2005=_USD2005,
    _2006=_USD2006,
    _2007=_USD2007,
    _2008=_USD2008,
    _2009=_USD2009,
    _2010=_USD2010,
    _2011=_USD2011,
    _2012=_USD2012,
    _2013=_USD2013,
    _2014=_USD2014,
    _2015=_USD2015,
    _2016=_USD2016,
    _2017=_USD2017,
    _2018=_USD2018,
    _2019=_USD2019,
    _2020=_USD2020,
    _2021=_USD2021,
    _2022=_USD2022,
    _2023=_USD2023,
    _2024=_USD2024,
    _2025=_USD2025,
)
"""Currency units for the United States Dollar (USD).

USD is the base currency in the currency units system so annual currency values are defined relative
to the value of USD in 2025 using annual inflation data from the
[US Inflation Calculator](https://www.usinflationcalculator.com/inflation/current-inflation-rates/).

"""

GBP = AnnualCurrencyUnit(
    name="GBP",
    _2000=_GBP2000,
    _2001=_GBP2001,
    _2002=_GBP2002,
    _2003=_GBP2003,
    _2004=_GBP2004,
    _2005=_GBP2005,
    _2006=_GBP2006,
    _2007=_GBP2007,
    _2008=_GBP2008,
    _2009=_GBP2009,
    _2010=_GBP2010,
    _2011=_GBP2011,
    _2012=_GBP2012,
    _2013=_GBP2013,
    _2014=_GBP2014,
    _2015=_GBP2015,
    _2016=_GBP2016,
    _2017=_GBP2017,
    _2018=_GBP2018,
    _2019=_GBP2019,
    _2020=_GBP2020,
    _2021=_GBP2021,
    _2022=_GBP2022,
    _2023=_GBP2023,
    _2024=_GBP2024,
    _2025=_GBP2025,
)
"""Currency units for the British Pound Sterling (GBP).

Annual currency values are defined relative to USD using annual exchange rate data from the
[Bank of England](https://www.bankofengland.co.uk/boeapps/database/).

"""

EUR = AnnualCurrencyUnit(
    name="EUR",
    _2000=_EUR2000,
    _2001=_EUR2001,
    _2002=_EUR2002,
    _2003=_EUR2003,
    _2004=_EUR2004,
    _2005=_EUR2005,
    _2006=_EUR2006,
    _2007=_EUR2007,
    _2008=_EUR2008,
    _2009=_EUR2009,
    _2010=_EUR2010,
    _2011=_EUR2011,
    _2012=_EUR2012,
    _2013=_EUR2013,
    _2014=_EUR2014,
    _2015=_EUR2015,
    _2016=_EUR2016,
    _2017=_EUR2017,
    _2018=_EUR2018,
    _2019=_EUR2019,
    _2020=_EUR2020,
    _2021=_EUR2021,
    _2022=_EUR2022,
    _2023=_EUR2023,
    _2024=_EUR2024,
    _2025=_EUR2025,
)
"""Currency units for the Eurozone Euro (EUR).

Annual currency values are defined relative to USD using annual exchange rate data from the
[Bank of England](https://www.bankofengland.co.uk/boeapps/database/).

"""

JPY = AnnualCurrencyUnit(
    name="JPY",
    _2000=_JPY2000,
    _2001=_JPY2001,
    _2002=_JPY2002,
    _2003=_JPY2003,
    _2004=_JPY2004,
    _2005=_JPY2005,
    _2006=_JPY2006,
    _2007=_JPY2007,
    _2008=_JPY2008,
    _2009=_JPY2009,
    _2010=_JPY2010,
    _2011=_JPY2011,
    _2012=_JPY2012,
    _2013=_JPY2013,
    _2014=_JPY2014,
    _2015=_JPY2015,
    _2016=_JPY2016,
    _2017=_JPY2017,
    _2018=_JPY2018,
    _2019=_JPY2019,
    _2020=_JPY2020,
    _2021=_JPY2021,
    _2022=_JPY2022,
    _2023=_JPY2023,
    _2024=_JPY2024,
    _2025=_JPY2025,
)
"""Currency units for the Japanese Yen (JPY).

Annual currency values are defined relative to USD using annual exchange rate data from the
[Bank of England](https://www.bankofengland.co.uk/boeapps/database/).

"""

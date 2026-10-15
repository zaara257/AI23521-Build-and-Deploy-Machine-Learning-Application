"""Festival / holiday calendar used for the demand features.

The training data covers journeys between 1 Mar 2019 and 27 Jun 2019, so this is a
hand-compiled list of the main all-India festivals and public holidays that fall in
that window. It is deliberately small and easy to edit: add a (date, name) pair and
re-run `python train.py` to see whether it helps.

NOTE: dates were compiled by hand for 2019 - double-check them against an official
holiday calendar before quoting them in a report.
"""
from datetime import date

FESTIVALS = {
    date(2019, 3, 4): "Maha Shivaratri",
    date(2019, 3, 21): "Holi",
    date(2019, 4, 6): "Ugadi / Gudi Padwa",
    date(2019, 4, 13): "Ram Navami / Baisakhi",
    date(2019, 4, 14): "Ambedkar Jayanti / Tamil New Year",
    date(2019, 4, 17): "Mahavir Jayanti",
    date(2019, 4, 19): "Good Friday",
    date(2019, 5, 1): "Labour Day",
    date(2019, 5, 18): "Buddha Purnima",
    date(2019, 6, 5): "Eid al-Fitr",
}

# Journeys within this many days of a festival count as "near a festival".
FESTIVAL_WINDOW_DAYS = 3

# Indian school summer holidays: treated as the summer peak season.
SUMMER_PEAK_START = date(2019, 5, 1)


def nearest_festival(d: date):
    """Return (days_away, name) for the festival closest to date `d`."""
    best = min(FESTIVALS, key=lambda f: abs((f - d).days))
    return abs((best - d).days), FESTIVALS[best]

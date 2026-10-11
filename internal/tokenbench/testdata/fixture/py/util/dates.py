"""Date helpers shared by billing and reports."""

from datetime import date, datetime, timedelta


def parse_iso_date(text):
    """Parse YYYY-MM-DD into a date."""
    return datetime.strptime(text, "%Y-%m-%d").date()


def business_days_between(start, end):
    """Count weekdays from start up to but not including end."""
    days = 0
    current = start
    while current < end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


def end_of_month(day):
    """Last calendar day of the month containing day."""
    first_next = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return first_next - timedelta(days=1)

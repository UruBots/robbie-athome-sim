"""Validation for ROS domains reserved for isolated simulation lab runs."""


def validate_lab_domain(value):
    """Return a valid DDS lab domain, rejecting default and real-robot domains."""
    try:
        domain = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("a numeric lab ROS_DOMAIN_ID is required") from exc
    if not 1 <= domain <= 232 or domain == 10:
        raise ValueError("lab ROS_DOMAIN_ID must be in 1..232, excluding robot domain 10")
    return domain

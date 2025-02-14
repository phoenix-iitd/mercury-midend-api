import logging


def add_correlation_id(
    msg: str, correlation_id: str = "NO_CORR_ID", level: int = logging.INFO
) -> str:
    if level == logging.DEBUG:
        color = "\033[90m"  # gray
    elif level == logging.INFO:
        color = "\033[32m"  # green
    elif level == logging.WARNING:
        color = "\033[38;5;208m"  # orange
    elif level == logging.ERROR:
        color = "\033[31m"  # red
    else:
        color = ""
    reset = "\033[0m"
    return f"{color}[{correlation_id}]{reset} {msg}"

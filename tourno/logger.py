import contextvars
import functools
import logging
import re
import sys

### CLI Logging ###
_LOGGER_NAME = "swebench_experiments"
_DATE_FORMAT = "%H:%M:%S"
_PLAIN_FORMAT = "[%(asctime)s] [%(levelname)5s] [%(log_id)s] %(message)s"
_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[91m",
    logging.CRITICAL: "\033[35m",
}
_ID_PALETTE = [
    "\033[38;5;75m",
    "\033[38;5;114m",
    "\033[38;5;180m",
    "\033[38;5;139m",
    "\033[38;5;109m",
    "\033[38;5;216m",
    "\033[38;5;151m",
    "\033[38;5;182m",
    "\033[38;5;223m",
    "\033[38;5;116m",
    "\033[38;5;174m",
    "\033[38;5;143m",
]

_current_log_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_log_id", default=None
)


class _ColorFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(datefmt=_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        log_id = getattr(record, "log_id", "main")
        lc = _LEVEL_COLORS.get(record.levelno, "")
        ic = _ID_PALETTE[hash(log_id) % len(_ID_PALETTE)]
        ts = self.formatTime(record, self.datefmt)
        msg = (
            f"{ic}[{ts}]{_RESET} "
            f"{lc}[{record.levelname:>5s}]{_RESET} "
            f"{ic}[{log_id}]{_RESET} "
            f"{record.getMessage()}"
        )
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg += "\n" + record.exc_text
        if record.stack_info:
            msg += "\n" + self.formatStack(record.stack_info)
        return msg


class PrefixFilter(logging.Filter):
    def __init__(self, pattern: str):
        super().__init__()
        self._re = re.compile(pattern)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return bool(self._re.search(getattr(record, "log_id", "main")))


class _IdAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["log_id"] = self.extra["log_id"]
        return msg, kwargs


def setup(level: int = logging.INFO, filter_pattern: str | None = None, color: bool = True) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if color:
        handler.setFormatter(_ColorFormatter())
    else:
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT))
    if filter_pattern is not None:
        handler.addFilter(PrefixFilter(filter_pattern))
    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(suffix: str | None = None) -> logging.LoggerAdapter:
    ctx = _current_log_id.get()
    if suffix:
        log_id = f"{ctx}:{suffix}" if ctx else suffix
        _current_log_id.set(log_id)
    else:
        log_id = ctx or "main"
    return _IdAdapter(logging.getLogger(_LOGGER_NAME), {"log_id": log_id})


def log_metrics(metrics: dict[str, object], step: int | None = None) -> None:
    from rich.console import Console
    from rich.table import Table

    if not metrics:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green")
    if step is not None:
        table.title = f"Step {step}"

    for key, value in sorted(metrics.items()):
        value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        table.add_row(key, value_str)

    console = Console(stderr=True)
    with console.capture() as capture:
        console.print(table)

    get_logger().info("\n" + capture.get().rstrip())


def trace(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        token = _current_log_id.set(_current_log_id.get())
        try:
            return await fn(*args, **kwargs)
        finally:
            _current_log_id.reset(token)

    return wrapper


def init_weave(project_name: str) -> None:
    import weave

    weave.init(project_name)
    get_logger().info(f"Weave tracing enabled: project={project_name}")

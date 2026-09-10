"""Allow the CLI to run as ``python -m src.cli``."""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())

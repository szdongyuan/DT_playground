import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.license_validation import calculate_license_hash, get_mac_address


def get_default_output_dir() -> Path:
    """Return the executable directory when frozen, otherwise the project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PROJECT_ROOT


def generate_salt(output_dir: str | Path | None = None) -> Path:
    """Generate a salt file for the current machine."""
    target_dir = Path(output_dir or get_default_output_dir()).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    salt_path = target_dir / "salt"
    salt_path.write_text(
        calculate_license_hash(get_mac_address()),
        encoding="utf-8",
    )
    return salt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the machine-bound salt license file."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory in which to create the salt file (default: executable directory).",
    )
    args = parser.parse_args()
    salt_path = generate_salt(args.output_dir)
    print(f"Salt file generated: {salt_path}")


if __name__ == "__main__":
    main()

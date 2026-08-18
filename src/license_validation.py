"""Machine-bound license validation helpers."""

import ctypes
import hashlib
import hmac
import socket
from ctypes import wintypes
from pathlib import Path


MAX_ADAPTER_ADDRESS_LENGTH = 8
GAA_FLAG_INCLUDE_PREFIX = 0x0010
ERROR_BUFFER_OVERFLOW = 111
IF_TYPE_ETHERNET_CSMACD = 6
IF_TYPE_IEEE80211 = 71
LICENSE_SUFFIX = "dongyuan"

EXCLUDE_KEYWORDS = (
    "virtual",
    "hyper-v",
    "vmware",
    "virtualbox",
    "loopback",
    "teredo",
    "isatap",
    "bluetooth",
    "tap-windows",
    "microsoft kernel",
)

DESCRIPTION_KEYWORDS = (
    "realtek",
    "intel",
    "broadcom",
    "amd",
    "ethernet",
    "gigabit",
    "lan",
    "network",
    "integrated",
    "pci",
    "pcie",
    "pci-e",
)


class SOCKADDR(ctypes.Structure):
    _fields_ = [
        ("sa_family", wintypes.USHORT),
        ("sa_data", ctypes.c_char * 14),
    ]


class IP_ADAPTER_ADDRESSES(ctypes.Structure):
    pass


IP_ADAPTER_ADDRESSES._fields_ = [
    ("Length", wintypes.ULONG),
    ("IfIndex", wintypes.DWORD),
    ("Next", ctypes.POINTER(IP_ADAPTER_ADDRESSES)),
    ("AdapterName", ctypes.c_char_p),
    ("FirstUnicastAddress", ctypes.c_void_p),
    ("FirstAnycastAddress", ctypes.c_void_p),
    ("FirstMulticastAddress", ctypes.c_void_p),
    ("FirstDnsServerAddress", ctypes.c_void_p),
    ("DnsSuffix", ctypes.c_wchar_p),
    ("Description", ctypes.c_wchar_p),
    ("FriendlyName", ctypes.c_wchar_p),
    ("PhysicalAddress", wintypes.BYTE * MAX_ADAPTER_ADDRESS_LENGTH),
    ("PhysicalAddressLength", wintypes.DWORD),
    ("Flags", wintypes.DWORD),
    ("Mtu", wintypes.DWORD),
    ("IfType", wintypes.DWORD),
    ("OperStatus", ctypes.c_int),
]


def get_mac_address() -> str:
    """Return the first physical Ethernet or Wi-Fi adapter MAC address."""
    iphlpapi = ctypes.WinDLL("iphlpapi.dll")
    buffer_size = wintypes.ULONG(0)
    result = iphlpapi.GetAdaptersAddresses(
        socket.AF_UNSPEC,
        GAA_FLAG_INCLUDE_PREFIX,
        None,
        None,
        ctypes.byref(buffer_size),
    )
    if result != ERROR_BUFFER_OVERFLOW:
        raise ctypes.WinError(result)

    buffer = ctypes.create_string_buffer(buffer_size.value)
    adapter = ctypes.cast(buffer, ctypes.POINTER(IP_ADAPTER_ADDRESSES))
    result = iphlpapi.GetAdaptersAddresses(
        socket.AF_UNSPEC,
        GAA_FLAG_INCLUDE_PREFIX,
        None,
        adapter,
        ctypes.byref(buffer_size),
    )
    if result != 0:
        raise ctypes.WinError(result)

    while adapter:
        adapter_info = adapter.contents
        description = (adapter_info.Description or "").lower()
        friendly_name = (adapter_info.FriendlyName or "").lower()
        is_excluded = any(
            keyword in description or keyword in friendly_name
            for keyword in EXCLUDE_KEYWORDS
        )
        is_physical_description = any(
            keyword in description for keyword in DESCRIPTION_KEYWORDS
        )
        is_supported_type = adapter_info.IfType in (
            IF_TYPE_ETHERNET_CSMACD,
            IF_TYPE_IEEE80211,
        )

        if (
            adapter_info.PhysicalAddressLength > 0
            and is_supported_type
            and not is_excluded
            and is_physical_description
        ):
            return ":".join(
                f"{byte:02x}"
                for byte in adapter_info.PhysicalAddress[
                    : adapter_info.PhysicalAddressLength
                ]
            )

        adapter = adapter_info.Next

    raise RuntimeError("No physical MAC address found")


def calculate_license_hash(mac_address: str) -> str:
    """Calculate the license hash for a normalized MAC address."""
    return hashlib.sha256(
        f"{mac_address}{LICENSE_SUFFIX}".encode("utf-8")
    ).hexdigest()


def validate_license(app_root: str | Path) -> bool:
    """Validate the root salt file against this machine's MAC address."""
    try:
        stored_hash = (Path(app_root) / "salt").read_text(encoding="utf-8").strip().lower()
        mac_address = get_mac_address()
    except (OSError, UnicodeError, RuntimeError):
        return False

    expected_hash = calculate_license_hash(mac_address)
    return hmac.compare_digest(stored_hash, expected_hash)

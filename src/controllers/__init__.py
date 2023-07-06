REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .basic_controller_gpi import BasicMACGPI

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_gpi"] = BasicMACGPI
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
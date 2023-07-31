from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
from basic_controller_gpi import BasicMACGPI

# This multi-agent controller shares parameters between agents
class BasicMACGPIDiscrete(BasicMACGPI):
    pass
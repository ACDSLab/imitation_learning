#!/usr/bin/env python
"""Base class for system interfaces."""
from abc import ABCMeta, abstractmethod


class Environment(object):
    """Environment/system interface base class."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def wait_for_rollout(self, autonomous_control):
        """Wait until ready to do a rollout then return the latest data."""
        return None
    
    @abstractmethod
    def step(self, action=None):
        """Apply the indicated action and return the corresponding observations, rewards, and status."""
        return (None, None, True)

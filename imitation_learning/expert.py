#!/usr/bin/env python
"""Base class for experts."""
from abc import ABCMeta, abstractmethod


class Expert(object):
    """Expert base class."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def control_callback(self, msg):
        """Save the latest control command."""
        return [None, None]

    @abstractmethod
    def cost_callback(self, msg):
        """Save the latest cost."""
        return None

    @abstractmethod
    def status_callback(self, msg):
        """Save the latest status."""
        return None

    @abstractmethod
    def autonomous(self):
        """Return the autonomous mode"""
        return None

    @abstractmethod
    def action(self, obs):
        """Return the latest control command from observation."""
        return None

    @abstractmethod
    def ready(self):
        """Return the expert status"""
        return None

    @abstractmethod
    def cost(self):
        """Return the latest cost"""
        return None

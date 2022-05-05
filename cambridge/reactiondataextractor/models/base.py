# -*- coding: utf-8 -*-
"""
Base
=====

This module contains the base extraction class

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
from abc import abstractmethod
import logging

log = logging.getLogger(__name__)


class BaseExtractor:

    @abstractmethod
    def extract(self):
        """This method extracts objects (arrows, conditions, diagrams or labels) from ``fig``"""
        pass

    @abstractmethod
    def plot_extracted(self, ax):
        """This method places extracted objects on canvas of ``ax``"""
        pass

    @property
    @abstractmethod
    def extracted(self):
        """This method returns extracted objects"""
        pass

# -*- coding: utf-8 -*-
"""
Diagrams
=======

This module contains a single arrow extraction class.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import copy
import logging
from matplotlib.patches import Rectangle
import numpy as np

from scipy.ndimage import label
from skimage.transform import probabilistic_hough_line

from ..actions import skeletonize
from ..models import SolidArrow, BaseExtractor, NotAnArrowException, NoArrowsFoundException, Point
from ..models.segments import FigureRoleEnum
from ..utils.processing import approximate_line
from ..utils.processing import is_a_single_line
from .. import settings

log = logging.getLogger('extract.arrows')


class ArrowExtractor(BaseExtractor):
    """Main class for extracting reaction arrows

    :param fig: main figure
    :type fig: Fig
    :param min_arrow_length: minimum length of a detected line to accept as a valid arrow candidate
    :type min_arrow_length: int"""

    def __init__(self, fig=None, min_arrow_length=None):
        if fig is None:
            self.fig = settings.main_figure[0]
        if min_arrow_length is None:
            self.min_arrow_length = int(self.fig.single_bond_length)
        self._extracted = None

    def extract(self, fig=None):
        """Main extraction method"""
        self._extracted = self.find_arrows()
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        if not self.extracted:
            pass
        else:
            for arrow in self.extracted:
                panel = arrow.panel
                rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                      facecolor='y', edgecolor=None, alpha=0.4)
                ax.add_patch(rect_bbox)

    def find_arrows(self):
        """
        Arrow finding algorithm.

        Finds lines of length at least ``min_arrow length`` in ``fig`` and detects arrows
        using a rule-based algorithm. Can be extended to find other types of arrows.
        :return: collection of found arrows
        :rtype: list
        """
        threshold = self.min_arrow_length//2
        arrows = self.find_solid_arrows(threshold=threshold)

        if not arrows:
            log.warning('No arrows have been found in the image')
            raise NoArrowsFoundException('No arrows have been found; aborting.')

        return list(set(arrows))

    def find_solid_arrows(self, threshold):
        """
        Finds all solid (straight) arrows in ``fig`` subject to ``threshold`` number of pixels and ``min_arrow_length``
        minimum line length.
        :param int threshold: threshold number of pixels needed to define a line (Hough transform param).
        :return: collection of arrow objects
        :rtype: list
        """
        def inrange(cc, point):
            """Returns True if a ``point`` lies inside ``cc``, else return False."""
            return point.row in range(cc.top, cc.bottom+1) and point.col in range(cc.left, cc.right+1)

        fig = self.fig
        img = copy.deepcopy(fig.img)

        arrows = []
        skeletonized = skeletonize(fig)
        all_lines = probabilistic_hough_line(skeletonized.img, threshold=threshold,
                                             line_length=self.min_arrow_length, line_gap=3)

        for line in all_lines:
            points = [Point(row=y, col=x) for x, y in line]
            # Choose one of points to find the label and pixels in the image
            p1, p2 = points
            labelled_img, _ = label(img)
            p1_label = labelled_img[p1.row, p1.col]
            p2_label = labelled_img[p2.row, p2.col]
            if p1_label != p2_label:  # Hough transform can find lines spanning several close ccs; these are discarded
                log.debug('A false positive was found when detecting a line. Discarding...')
                continue
            else:
                parent_label = labelled_img[p1.row, p1.col]

                parent_panel = [cc for cc in fig.connected_components if inrange(cc, p1) and inrange(cc, p2)][0]

            # Break the line down and check whether it's a single line
            if not is_a_single_line(skeletonized, parent_panel, self.min_arrow_length//2):
                continue

            arrow_pixels = np.nonzero(labelled_img == parent_label)
            arrow_pixels = list(zip(*arrow_pixels))
            try:
                new_arrow = SolidArrow(arrow_pixels, line=approximate_line(p1, p2), panel=parent_panel)
            except NotAnArrowException as e:
                log.info('An arrow candidate was discarded - ' + str(e))
            else:
                arrows.append(new_arrow)
                parent_cc = [cc for cc in fig.connected_components if cc == new_arrow.panel][0]
                parent_cc.role = FigureRoleEnum.ARROW

        return list(set(arrows))

# -*- coding: utf-8 -*-
"""
Diagrams
=======

This module contains a single diagram extraction class.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import copy
import logging
from matplotlib.patches import Rectangle
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import probabilistic_hough_line


from ..models import BaseExtractor, Rect, Figure, FigureRoleEnum, Panel
from ..utils.processing import dilate_fragments, erase_elements, isolate_patches, skeletonize_area_ratio, skeletonize
from .. import settings

log = logging.getLogger('extract.diagrams')


class DiagramExtractor(BaseExtractor):
    """Main class for extracting diagrams from chemical reaction schemes
    :param fig: main figure
    :type fig: Figure
    :param arrows: all arrows in the reaction scheme
    :type arrows: list[SolidArrow]"""
    def __init__(self, fig=None, arrows=None):
        self.fig = fig if fig is not None else settings.main_figure[0]
        self.arrows = arrows if arrows is not None else []
        self._extracted = None
        self.backbones = None

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def extract(self):
        """Main extraction method"""
        self.backbones = self.detect_backbones()
        self.fig.kernel_sizes = self._find_optimal_dilation_ksize()
        self._extracted = self.complete_structures()
        return self.extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        if not self.extracted:
            pass
        else:
            for panel in self.extracted:
                rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                      facecolor=(52/255, 0, 103/255), edgecolor=(6/255, 0, 99/255), alpha=0.4)
                ax.add_patch(rect_bbox)

    def detect_backbones(self):
        """
        Detects carbon backbones based on features such as size, aspect ratio and number of detected single bonds.

        Based on an estimated single bond length, each connected component is analysed to find the number of bonds.
        Area and aspect ratio are also used as features. Finally, a DBSCAN is performed on the formed, normalised
        dataset.
        :return: connected components classified as structural backbones
        :rtype: list
        """
        fig = self.fig
        ccs = fig.connected_components
        ccs = sorted(ccs, key=lambda panel: panel.area, reverse=True)
        cc_lines = []
        for cc in ccs:
            isolated_cc_fig = isolate_patches(fig, [cc])
            isolated_cc_fig = skeletonize(isolated_cc_fig)

            num_lines = len(probabilistic_hough_line(isolated_cc_fig.img,
                                                     line_length=fig.single_bond_length, threshold=10, line_gap=0))
            cc_lines.append(num_lines)

        cc_lines = np.array(cc_lines).reshape(-1, 1)
        area = np.array([cc.area for cc in ccs]).reshape(-1, 1)
        aspect_ratio = np.array([cc.aspect_ratio for cc in ccs]).reshape(-1, 1)
        mean_area = np.mean(area)

        data = np.hstack((cc_lines, area, aspect_ratio))
        data = MinMaxScaler().fit_transform(data)

        labels = DBSCAN(eps=0.15, min_samples=20).fit_predict(data)

        paired = list(zip(ccs, labels))
        paired = [(cc, label) if cc.area > mean_area else (cc, 0) for cc, label in paired]

        backbones = [panel for panel, label in paired if label == -1]
        backbones = set(backbones)
        arrows = set([cc for cc in fig.connected_components if cc.role == FigureRoleEnum.ARROW])
        backbones = list(backbones.difference(arrows))

        [setattr(backbone, 'role', FigureRoleEnum.STRUCTUREBACKBONE) for backbone in backbones]

        return backbones

    def complete_structures(self):
        """
        Dilates a figure and uses backbones to find complete chemical structures (backbones + superatoms etc.).

        Arrows are first removed to increase accuracy of the process. Figure is dilates around each backbone according
        to density of features around it. The diagrams are derived from the dilated backbones. Roles are assigned
        to the disconnected diagram parts.
        :return:bounding boxes of chemical structures
        :rtype: list
        """
        fig = self.fig
        fig_no_arrows = erase_elements(fig, self.arrows)
        dilated_structure_panels, other_ccs = self.find_dilated_structures(fig_no_arrows)
        structure_panels = self._complete_structures(dilated_structure_panels)
        self._assign_backbone_auxiliaries(structure_panels, other_ccs)  # Assigns cc roles
        temp = copy.deepcopy(structure_panels)
        # simple filtering to account for multiple backbone parts (disconnected by heteroatom characters)
        # corresponding to the same diagram
        for panel1 in temp:
            for panel2 in temp:
                if panel2.contains(panel1) and panel2 != panel1:
                    try:
                        structure_panels.remove(panel1)
                    except ValueError:
                        pass

        return list(set(structure_panels))

    def find_dilated_structures(self, fig=None):
        """
        Finds dilated structures by first dilating the image several times using backbone-specific kernel size.

        For each backbone, the figure is dilated using a backbone-specific kernel size. Dilated structure panel is then
        found based on comparison with the original backbone. A crop is made for each structure. If there is more than
        one connected component that is fully contained within the crop, it is noted and this information used later
        when the small disconnected ccs are assigned roles (This additional connected component is likely a label).
        :param Figure fig: Analysed figure
        :return: (dilated_structure_panels, other_ccs) pair of collections containing the dilated panels and
        separate ccs present within these dilated panels
        :rtype: tuple of lists
        """
        if fig is None:
            fig = self.fig
        dilated_structure_panels = []
        other_ccs = []
        dilated_imgs = {}

        for backbone in self.backbones:
            ksize = fig.kernel_sizes[backbone]
            try:
                dilated_temp = dilated_imgs[ksize]
            except KeyError:
                dilated_temp = dilate_fragments(fig, ksize)
                dilated_imgs[ksize] = dilated_temp

            dilated_structure_panel = [cc for cc in dilated_temp.connected_components if cc.contains(backbone)][0]
            # Crop around with a small extension to get the connected component correctly
            structure_crop = dilated_structure_panel.create_extended_crop(dilated_temp, extension=5)
            other = [structure_crop.in_main_fig(c) for c in structure_crop.connected_components if
                     structure_crop.in_main_fig(c) != dilated_structure_panel]
            other_ccs.extend(other)
            dilated_structure_panels.append(dilated_structure_panel)

        return dilated_structure_panels, other_ccs

    def _assign_backbone_auxiliaries(self, structure_panels, cno_ccs):
        """
        Assigns roles to small disconnected diagram parts.

        Takes in the detected structures panels and ccs that are contained inside structure panels but are
        non-overlapping  (``cno_ccs``) - including in the dilated figure. Assigns roles to all (small) connected
        components contained within structure panels, and finally resets role for the special ``cno_ccs``. These are
        likely to be labels lying very close to the diagrams themselves.
        :param [Panel,...] structure_panels: iterable of found structure panels
        :param [Panel,...] cno_ccs: contained-non-overlapping cc;ccs that are not parts of diagrams even though
        their panels are situated fully inside panels of chemical diagrams (common with labels).
        :return: None (mutates ''role'' attribute of each relevant connected component)
        """
        fig = self.fig

        for parent_panel in structure_panels:
            for cc in fig.connected_components:
                if parent_panel.contains(cc):  # Set the parent panel for all
                    setattr(cc, 'parent_panel', parent_panel)
                    if cc.role != FigureRoleEnum.STRUCTUREBACKBONE:
                        # Set role for all except backbone which had been set
                        setattr(cc, 'role', FigureRoleEnum.STRUCTUREAUXILIARY)

        for cc in cno_ccs:
            # ``cno_ccs`` are dilated - find raw ccs in ``fig``
            fig_ccs = [fig_cc for fig_cc in fig.connected_components if cc.contains(fig_cc)]

            [setattr(fig_cc, 'role', None) for fig_cc in fig_ccs]

        log.debug('Roles of structure auxiliaries have been assigned.')

    def _complete_structures(self, dilated_structure_panels):
        """Uses ``dilated_structure_panels`` to find all constituent ccs of each chemical structure.

        Finds connected components belonging to a chemical structure and creates a large panel out of them. This
        effectively normalises panel sizes to be independent of chosen dilation kernel sizes.
        :return [Panel,...]: iterable of Panels bounding complete chemical structures.
        """

        structure_panels = []
        for dilated_structure in dilated_structure_panels:
            constituent_ccs = [cc for cc in self.fig.connected_components if dilated_structure.contains(cc)]
            parent_structure_panel = Panel.create_megarect(constituent_ccs)
            structure_panels.append(parent_structure_panel)
        return structure_panels

    def _find_optimal_dilation_ksize(self):
        """
        Use structural backbones to calculate local skeletonised-pixel ratio and find optimal dilation kernel sizes for
        structural segmentation. Each backbone is assigned its own dilation kernel to account for varying skel-pixel
        ratio around different backbones
        :return: kernel sizes appropriate for each backbone
        :rtype: dict
        """

        backbones = [cc for cc in self.fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE]

        kernel_sizes = {}
        for backbone in backbones:
            left, right, top, bottom = backbone
            horz_ext, vert_ext = backbone.width // 2, backbone.height // 2
            crop_rect = Rect(left - horz_ext, right + horz_ext, top - vert_ext, bottom + vert_ext)
            p_ratio = skeletonize_area_ratio(self.fig, crop_rect)
            log.debug(f'found in-crop skel_pixel ratio: {p_ratio}')

            if p_ratio >= 0.02:
                kernel_size = 4
            elif 0.01 < p_ratio < 0.02:
                kernel_size = np.ceil(20 - 800 * p_ratio)
            else:
                kernel_size = 12
            kernel_sizes[backbone] = kernel_size

        log.debug(f'Structure segmentation kernels:{kernel_sizes.values()}')
        return kernel_sizes

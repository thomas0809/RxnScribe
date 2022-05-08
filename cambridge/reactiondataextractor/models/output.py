# -*- coding: utf-8 -*-
"""
Output
=======

This module contains classes used for representing the output of extraction procedures.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

from sklearn.cluster import DBSCAN

from .reaction import Diagram, ReactionStep
from .segments import ReactionRoleEnum
from .utils import Point, PrettyFrozenSet, PrettyList
from ..actions import find_nearby_ccs, extend_line
from .reaction import Conditions
from .. import settings


class Graph(ABC):
    """
    Generic directed graph class

    :param graph_dict: underlying graph mapping
    :type graph_dict: dict
    """

    def __init__(self, graph_dict=None):
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

    @abstractmethod
    def _generate_edge(self, *args) :
        """
        This method should update the graph dict with a connection between vertices,
        possibly adding some edge annotation.
        """
        return NotImplemented

    @abstractmethod
    def edges(self):
        """
        This method should return all edges (partially via invoking the `_generate_edge` method).
        """
        return NotImplemented

    @abstractmethod
    def __str__(self):
        """
        A graph needs to have a __str__ method to constitute a valid output representation.
        """
    @property
    def nodes(self):
        return self._graph_dict.keys()

    def add_vertex(self, vertex):
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def find_isolated_vertices(self):
        """
        Returns all isolated vertices. Can be used for output validation
        :return: collection of isolated (unconnected) vertices
        """
        graph = self._graph_dict
        return [key for key in graph if graph[key] == []]

    def find_path(self, vertex1, vertex2, path=None):
        if path is None:
            path = []
        path += [vertex1]
        graph = self._graph_dict
        if vertex1 not in graph:
            return None
        if vertex2 in graph[vertex1]:
            return path + [vertex2]
        else:
            for value in graph[vertex1]:
                return self.find_path(value, vertex2, path)


class ReactionScheme(Graph):
    """Main class used for representing the output of an extraction process

    :param conditions: all extracted reaction conditions
    :type conditions: list[Conditions]
    :param diags: all extracted chemical diagrams
    :type diags: list[Diagram]
    :param fig: Analysed figure
    :type fig: Figure"""
    def __init__(self, conditions, diags, fig):
        self._conditions = conditions
        self._diags = diags
        super().__init__()
        self._reaction_steps = ([self._scan_form_reaction_step(step_conditions)
                                           for step_conditions in conditions])
        self._pretty_reaction_steps = PrettyList(self._reaction_steps)
        self.create_graph()
        self._start = None  # start node(s) in a graph
        self._end = None   # end node(s) in a graph
        self._fig = fig
        graph = self._graph_dict
        self.set_start_end_nodes()

        self._single_path = True if len(self._start) == 1 and len(self._end) == 1 else False

    def edges(self):
        if not self._graph_dict:
            self.create_graph()

        return {k: v for k, v in self._graph_dict.items()}

    def _generate_edge(self, key, successor):

        self._graph_dict[key].append(successor)

    def __repr__(self):
        return f'ReactionScheme({self._reaction_steps})'

    def __str__(self):
        # if self._single_path:
        #     path = self.find_path(self.reactants, self.products)
        #     return '  --->  '.join((' + '.join(str(species) for species in group)) for group in path)
        # else:
        return '\n'.join([str(reaction_step) for reaction_step in self._reaction_steps])

    def __eq__(self, other):
        if isinstance(other, ReactionScheme):
            return other._graph_dict == self._graph_dict
        return False

    @property
    def reaction_steps(self):
        return self._reaction_steps

    @property
    def graph(self):
        return self._graph_dict

    @property
    def reactants(self):
        return self._start

    @property
    def products(self):
        return self._end

    def long_str(self):
        """Longer str method - contains more information (eg conditions)"""
        return f'{self._reaction_steps}'

    def draw_segmented(self, out=False):
        """Draw the segmented figure. If ``out`` is True, the figure is returned and can be saved"""
        y_size, x_size = self._fig.img.shape
        f, ax = plt.subplots(figsize=(x_size/100, y_size/100))
        ax.imshow(self._fig.img, cmap=plt.cm.binary)
        params = {'facecolor': 'g', 'edgecolor': None, 'alpha': 0.3}
        for step_conditions in self._conditions:
            panel = step_conditions.arrow.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='y', edgecolor=None, alpha=0.4)
            ax.add_patch(rect_bbox)

            for t in step_conditions.text_lines:
                panel = t.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)
            # for panel in step_conditions.structure_panels:
            #     rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
            #                           panel.bottom - panel.top, **params)
            #     ax.add_patch(rect_bbox)

        params = {'facecolor': (66 / 255, 93 / 255, 166 / 255),
                  'edgecolor': (6 / 255, 33 / 255, 106 / 255),
                  'alpha': 0.4}
        for diag in self._diags:
            panel = diag.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor=(52/255, 0, 103/255), edgecolor=(6/255, 0, 99/255), alpha=0.4)
            ax.add_patch(rect_bbox)
            if diag.label:
                panel = diag.label.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)
        ax.axis('off')
        if out:
            return f
        else:
            plt.show()

    def create_graph(self):
        """
        Unpack reaction steps to create a graph from individual steps
        :return: completed graph dictionary
        """
        graph = self._graph_dict
        for step in self._reaction_steps:
            [self.add_vertex(PrettyFrozenSet(species_group)) for species_group in step]
            self.add_vertex(step.conditions)

        for step in self._reaction_steps:
            self._generate_edge(step.reactants, step.conditions)
            self._generate_edge(step.conditions, step.products)

        return graph

    def set_start_end_nodes(self):
        """
        Finds and return the first vertex in a graph (group of reactants). Unpack all groups from ReactionSteps into
        a Counter. The first group is a group that is counted only once and exists as a key in the graph dictionary.
        Other groups (apart from the ultimate products) are counted twice (as a reactant in one step and a product in
        another).
        """
        group_count = Counter(group for step in self._reaction_steps for group in (step.reactants, step.products))
        self._start = [group for group, count in group_count.items() if count == 1 and
                       all(species.role == ReactionRoleEnum.STEP_REACTANT for species in group)]

        self._end = [group for group, count in group_count.items() if count == 1 and
                     all(species.role == ReactionRoleEnum.STEP_PRODUCT for species in group)]

    def find_path(self, group1, group2, path=None):
        """ Recursive routine for simple path finding between reactants and products"""
        graph = self._graph_dict
        if path is None:
            path = []
        path += [group1]
        if group1 not in graph:
            return None

        successors = graph[group1]
        if group2 in successors:
            return path+[group2]
        else:
            for prod in successors:
                return self.find_path(prod, group2, path=path)
        return None

    def to_json(self):
        # reactions = [self._json_generic_recursive(start_node) for start_node in self._start]
        json_dict = {}

        nodes = {label: node for label, node in zip(map(str, range(50)), self.nodes)}
        json_dict['node_labels'] = nodes
        adjacency = {}
        for node1, out_nodes in self.graph.items():
            node1_label = [label for label, node in nodes.items() if node == node1][0]
            out_nodes_labels = [label for label, node in nodes.items() if node in out_nodes]
            
            adjacency[node1_label] = out_nodes_labels
        json_dict['adjacency'] = adjacency

        for label, node in json_dict['node_labels'].items():
            if hasattr(node, '__iter__'):
                contents = []
                for diagram in node:
                    if diagram.label:
                        content = {'smiles': diagram.smiles, 'label': [sent.text.strip() for sent in diagram.label.text ]}
                    else:
                        content = {'smiles': diagram.smiles, 'label': None}
                    contents.append(content)
                json_dict['node_labels'][label] = contents
            elif isinstance(node, Conditions):
                contents = node.conditions_dct
                json_dict['node_labels'][label] = contents

        return json.dumps(json_dict, indent=4)

    # def _json_generic_recursive(self, start_key, json_obj=None):
    #     """
    #     Generic recursive json string generator. Takes in a single ``start_key`` node and builds up the ``json_obj`` by
    #     traverding the reaction graph
    #     :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
    #     :param json_obj: a dictionary created in the recursive procedure (ready for json dumps)
    #     :return:  dict; the created ``json_obj``
    #     """
    #     graph = self._graph_dict
    #
    #     if json_obj is None:
    #         json_obj = {}
    #
    #     node = start_key
    #
    #     if hasattr(node, '__iter__'):
    #         contents = [{'smiles': species.smiles, 'label': str(species.label)} for species in node]
    #     else:
    #         contents = str(node)   # Convert the conditions_dct directly
    #
    #     json_obj['contents'] = contents
    #     successors = graph[node]
    #     if not successors:
    #         json_obj['successors'] = None
    #         return json_obj
    #     else:
    #         json_obj['successors'] = []
    #         for successor in successors:
    #             json_obj['successors'].append(self._json_generic_recursive(successor))
    #
    #     return json_obj

    def to_smirks(self, start_key=None, species_strings=None):
        """
        Converts the reaction graph into a SMIRKS (or more appropriately - reaction SMILES, its subset). Also outputs
        a string containing auxiliary information from the conditions' dictionary.
        :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
        :param species_strings: list of found smiles strings (or chemical formulae) built up in the procedure and ready
        for joining into a single SMIRKS string.
        :return: (str, str) tuple containing a (reaction smiles, auxiliary info) pair
        """
        if not self._single_path:
            return NotImplemented  # SMIRKS only work for single-path reaction

        graph = self._graph_dict

        if start_key is None:
            start_key = self._start[0]

        if species_strings is None:
            species_strings = []

        node = start_key

        if hasattr(node, '__iter__'):  # frozenset of reactants or products
            species_str = '.'.join(species.smiles for species in node)
        else:  # Conditions object
            # The string is a sum of coreactants, catalysts (which have small dictionaries holding names and values/units)
            species_vals = '.'.join(species_dct['Species'] for group in iter((node.coreactants, node.catalysts))
                                    for species_dct in group)
            # and auxiliary species with simpler structures (no units)
            species_novals = '.'.join(group for group in node.other_species)
            species_str = '.'.join(filter(None, [species_vals, species_novals]))

        species_strings.append(species_str)

        successors = graph[node]
        if not successors:
            smirks ='>'.join(species_strings)
            return smirks
        else:
            return self.to_smirks(successors[0], species_strings)

        # return smirks, [node.conditions_dct for node in graph if isinstance(node, Conditions)]

    def _scan_form_reaction_step(self, step_conditions):
        """
        Scans an image around a single arrow to give reactants and products in a single reaction step
        :param Conditions step_conditions: Conditions object containing ``arrow`` around which the scan is performed
        :return: conditions and diagrams packed inside a reaction step
        :rtype: ReactionStep
        """
        arrow = step_conditions.arrow
        diags = self._diags

        endpoint1, endpoint2 = extend_line(step_conditions.arrow.line,
                                           extension=arrow.pixels[0].separation(arrow.pixels[-1]) * 0.75)
        react_side_point = step_conditions.arrow.react_side[0]
        endpoint1_close_to_react_side = endpoint1.separation(react_side_point) < endpoint2.separation(react_side_point)
        if endpoint1_close_to_react_side:
            react_endpoint, prod_endpoint = endpoint1, endpoint2
        else:
            react_endpoint, prod_endpoint = endpoint2, endpoint1

        initial_distance = 1.5 * np.sqrt(np.mean([diag.panel.area for diag in diags]))
        extended_distance = 4 * np.sqrt(np.mean([diag.panel.area for diag in diags]))
        distance_fn = lambda diag: 1.5 * np.sqrt(diag.panel.area)

        distances = initial_distance, distance_fn
        extended_distances = extended_distance, distance_fn
        reactants = find_nearby_ccs(react_endpoint, diags, distances,
                                    condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        if not reactants:
            reactants = find_nearby_ccs(react_endpoint, diags, extended_distances,
                                        condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        if not reactants:
            reactants = self._search_elsewhere('up-right', step_conditions.arrow, distances)

        products = find_nearby_ccs(prod_endpoint, diags, distances,
                                   condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)

        if not products:
            products = find_nearby_ccs(prod_endpoint, diags, extended_distances,
                                       condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)

        if not products:
            products = self._search_elsewhere('down-left', step_conditions.arrow, distances)

        [setattr(reactant, 'role', ReactionRoleEnum.STEP_REACTANT) for reactant in reactants]
        [setattr(product, 'role', ReactionRoleEnum.STEP_PRODUCT) for product in products]

        return ReactionStep(reactants, products, conditions=step_conditions)

    def _search_elsewhere(self, where, arrow, distances):
        """
        Looks for structures in a different line of a multi-line reaction scheme.

        If a reaction scheme ends unexpectedly either on the left or right side of an arrows (no species found), then
        a search is performed in the previous or next line of a reaction scheme respectively (assumes multiple lines
        in a reaction scheme). Assumes left-to-right reaction scheme. Estimates the optimal alternative search point
        using arrow and diagrams' coordinates in a DBSCAN search.
        This gives clusters corresponding to the multiple lines in a reaction scheme. Performs a search in the new spot.
        :param str where: Allows either 'down-left' to look below and to the left of arrow, or 'up-right' (above to the right)
        :param Arrow arrow: Original arrow, around which the search failed
        :param (float, lambda) distances: pair containing initial search distance and a distance function (usually same as
        in the parent function)
        :return: Collection of found species
        :rtype: list[Diagram]
        """
        assert where in ['down-left', 'up-right']
        fig = settings.main_figure[0]
        diags = self._diags

        X = np.array([s.center[1] for s in diags] + [arrow.panel.center[1]]).reshape(-1, 1)  # the y-coordinate
        eps = np.mean([s.height for s in diags])
        dbscan = DBSCAN(eps=eps, min_samples=2)
        y = dbscan.fit_predict(X)
        num_labels = max(y) - min(y) + 1  # include outliers (label -1) if any
        arrow_label = y[-1]
        clustered = []
        for val in range(-1, num_labels):
            if val == arrow_label:
                continue  # discard this cluster - want to compare the arrow with other clusters only
            cluster = [centre for centre, label in zip(X, y) if label == val]
            if cluster:
                clustered.append(cluster)
        centres = [np.mean(cluster) for cluster in clustered]
        centres.sort()
        if where == 'down-left':
            move_to_vertical = [centre for centre in centres if centre > arrow.panel.center[1]][0]
            move_to_horizontal = np.mean([structure.width for structure in diags])
        elif where == 'up-right':
            move_to_vertical = [centre for centre in centres if centre < arrow.panel.center[1]][-1]
            move_to_horizontal = fig.img.shape[1] - np.mean([structure.width for structure in diags])
        else:
            raise ValueError("'where' takes in one of two values : ('down-left', 'up-right') only")
        species = find_nearby_ccs(Point(move_to_vertical, move_to_horizontal), diags, distances)

        return species

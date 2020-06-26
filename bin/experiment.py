from plot_partition import Plotter
from distriopt import VirtualNetwork
from pathlib import Path
from distriopt.embedding.physical import PhysicalNetwork
import networkx as nx
from metis_converter import MetisConverter


class Experiment(object):
    def __init__(self, physical_topo, solution_dict, virtual_topo):
        self.physical_topo = physical_topo
        self.nx_physical_graph = physical_topo.g
        self.solution_dict = solution_dict
        empty_nodes = []
        for node in self.solution_dict.keys():
            if not self.solution_dict[node]:
                empty_nodes.append(node)
        for e in empty_nodes:
            del self.solution_dict[e]

        self.virtual_topo = virtual_topo

    def is_tree(self, nx_graph):
        return nx.is_connected(nx_graph) and (len(self.physical_topo.g.nodes()) - 1 == len(self.physical_topo.g.edges()))

    def node_used(self):
        return list(self.solution_dict.keys())

    def host_links_overcommitment(self):
        physical_hosts = self.physical_topo.compute_nodes
        links_commitment = {phost:0 for phost in physical_hosts}
        virtual_links = self.virtual_topo.edges()
        for e1, e2 in virtual_links:
            h1, h2 = self.find_physical_host(e1), self.find_physical_host(e2)
            if h1 != h2:
                path = self.physical_topo.find_path(h1,h2)
                if len(path) < 2:
                    raise RuntimeError(f"path between {h1}, {h2} is < 2")
                (n1, n2, _), (n3, n4, _) = path[0], path[-1]
                for u, v in [(n1, n2), (n3, n4)]:
                    if u in links_commitment.keys():
                        links_commitment[u] += self.virtual_topo.req_rate(e1, e2)
                    else:
                        links_commitment[v] += self.virtual_topo.req_rate(e1, e2)

        physical_links = self.physical_topo.edges()
        links_capacity={phost: 0 for phost in physical_hosts}
        for u, v in physical_links:
            if u in links_capacity.keys():
                links_capacity[u] += self.physical_topo.interfaces_ids(u, v)[0]["rate"]
            if v in links_capacity.keys():
                links_capacity[v] += self.physical_topo.interfaces_ids(u, v)[0]["rate"]

        return {phost: links_commitment[phost]-links_capacity[phost] for phost in physical_hosts}

    def total_links_overcommitment(self):
        if not self.is_tree(self.nx_physical_graph):
            raise RuntimeError("the physical network is not a Tree")
        links_commitment = {}
        virtual_links = self.virtual_topo.edges()
        for e1, e2 in virtual_links:
            h1, h2 = self.find_physical_host(e1), self.find_physical_host(e2)
            if h1 != h2:
                path = self.physical_topo.find_path(h1, h2)
                if len(path) < 2:
                    raise RuntimeError(f"path between {h1}, {h2} is < 2")
                for l1, l2, _ in path:
                    link = tuple(sorted([l1, l2]))
                    if link not in links_commitment:
                        links_commitment[link] = 0
                    links_commitment[link] += self.virtual_topo.req_rate(e1, e2)

        for link in links_commitment:
            real_rate = self.physical_topo.interfaces_ids(link[0], link[1])[0]["rate"]
            links_commitment[link] -= real_rate


        return links_commitment if links_commitment else {None:-9999999}



    def find_physical_host(self,vnode):
        for phost in self.solution_dict:
            for vnode_ in self.solution_dict[phost]:
                if vnode_ == vnode:
                    return phost

        raise RuntimeError(f"{vnode} is not assignes to any physical host")


    def cpu_overcommitment(self):
        used_cpu={}
        for pnode in self.solution_dict.keys():
            vnodes = self.solution_dict[pnode]
            cpu = 0
            for vnode in vnodes:
                cpu += self.virtual_topo.req_cores(vnode)
            used_cpu[pnode] = cpu

        return {pnode: used_cpu[pnode] - self.physical_topo.cores(pnode) for pnode in self.solution_dict.keys()}

    def memory_overcommitment(self):
        used_memory = {}
        for pnode in self.solution_dict.keys():
            vnodes = self.solution_dict[pnode]
            memory = 0
            for vnode in vnodes:
                memory += self.virtual_topo.req_memory(vnode)
            used_memory[pnode] = memory

        return {pnode: used_memory[pnode] - self.physical_topo.memory(pnode) for pnode in self.solution_dict.keys()}

import pickle

from distriopt.embedding.algorithms import (
    EmbedBalanced,
    EmbedILP,
    EmbedPartition,
    EmbedGreedy,
)


if __name__ == '__main__':
    physical = PhysicalNetwork.from_files(
        "/vagrant/bin/physical_instances/lyon")
    print(list(physical.compute_nodes))

    print(nx.is_connected(physical.g), (len(physical.g.nodes()),len(physical.g.edges())))
    #print(physical.find_path("Node3", "Node2"))

    virtual_topo = VirtualNetwork.create_fat_tree(k=4, density=2, req_cores=1, req_memory=1000, req_rate=1000)
    converter = MetisConverter(virtual_topo)
    mapping = converter.build_mapping(metis_graph_file="metis_graphs/fat_tree_k4_d2",
                                      metis_share_file="/vagrant/bin/metis_shares/lyon.txt")
    print(mapping)

    physical_names_mapping = {phy_name:metis_name for phy_name, metis_name in zip(list(physical.compute_nodes),mapping.keys())}
    print(physical_names_mapping)
    metis_name_mapping = {physical_names_mapping[x]: x for x in physical_names_mapping.keys()}
    print(metis_name_mapping)
    mapping_with_pyhisical_names = {metis_name_mapping[node]: mapping[node]for node in mapping.keys()}
    print(mapping_with_pyhisical_names)
    exp = Experiment(physical_topo=physical,solution_dict=mapping_with_pyhisical_names,virtual_topo=virtual_topo)
    print(exp.cpu_overcommitment())
    print(exp.memory_overcommitment())
    #print(physical.interfaces_ids('Node3', 'SW1'))
    print(exp.host_links_overcommitment())
    print(exp.total_links_overcommitment())




    #Plotter.plot_physical_network(physical, "/vagrant/bin/physical_instances/example1")

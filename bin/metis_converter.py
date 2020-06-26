from distriopt import VirtualNetwork
from pathlib import Path
import networkx as nx
import subprocess
from plot_partition import Plotter

import logging
logging.basicConfig(filename='.metis_converter.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

logging.info('metisConverter logging File')

class MetisConverter(object):
    def __init__(self, virtual_network):
        """
        MetisConverer takes in input the VirtualNetwork object from distriopt library and convert it a Maxinet input
        graph and ot creates a Metis input file
        :param virtual_network: VirtualNetwork obj from distriopt library
        """
        self.virtual_network = virtual_network
        self.metis_node_mapping = None
        self.node_metis_mapping = None
        self.edges=virtual_network.edges()
        self.nodes=virtual_network.nodes()
        self.edges_metis = None
        self.nodes_metis = None

    def convert_in_maxinet_dict(self):
        nodes = self.virtual_network.nodes()
        #print(nodes)
        maxinet_nodes = dict()

        for n in nodes:
            if not n.startswith("host"):
                maxinet_nodes[n] = {"weight": 1, "connected_switches": []}

        for n in maxinet_nodes.keys():
            connected_nodes = self.virtual_network.neighbors(n)
            for connected_node in connected_nodes.keys():
                if connected_node.startswith("host"):
                    maxinet_nodes[n]["weight"] += 1
                else:
                    maxinet_nodes[n]["connected_switches"].append(connected_node)

        return maxinet_nodes


    def convert_in_metis_dict(self, maxinet_dict):
        self.metis_node_mapping = {num+1: node for num, node in enumerate(maxinet_dict.keys())}
        self.node_metis_mapping = {self.metis_node_mapping[num]: num for num in self.metis_node_mapping.keys()}
        for node in self.node_metis_mapping:
            logging.debug(f"node_metis_mapping {node}: {self.node_metis_mapping[node]}")
        metis_dict = {num: {"weight": None, "edges": []} for num in self.metis_node_mapping.keys()}
        for node in maxinet_dict.keys():
            num = self.node_metis_mapping[node]
            metis_dict[num]["weight"] = maxinet_dict[node]["weight"]
            for neighboor in maxinet_dict[node]["connected_switches"]:
                neighboor_mapped = self.node_metis_mapping[neighboor]
                required_edge_rate = self.virtual_network.req_rate(node, neighboor)
                metis_dict[num]["edges"] += [neighboor_mapped, required_edge_rate]
        return metis_dict

    def get_metis_nodes(self):
        nodes = []
        for node in self.virtual_network.nodes():
            if not node.startswith("host"):
                nodes.append(node)

        return nodes

    def get_metis_edges(self):
        edges = []
        for u, v in self.virtual_network.edges():
            if not u.startswith("host") and not v.startswith("host"):
                edges.append((u, v))

        return edges

    def create_metis_file(self, metis_dict, path):
        nodes, edges = len(self.get_metis_nodes()), len(self.get_metis_edges())
        sorted_keys = sorted(list(metis_dict.keys()))
        metis_lines = [[nodes, edges, "011", "0"]]
        for k in sorted_keys:
            weight = metis_dict[k]["weight"]
            edges = metis_dict[k]["edges"]
            line = [weight] + edges
            metis_lines.append(line)

        with open(Path(path), "w") as file:
            for line in metis_lines:
                file.write(" ".join([str(x) for x in line]) + "\n")

        logging.debug(f"Metis Dict :")
        for key in metis_dict:
            logging.debug(f"{key} {metis_dict[key]}")
        logging.info(f"Metis file Created in path: {path}")
        return metis_lines

    def get_physical_hosts(self, share_path):
        with open(share_path, "r") as file:
            lines = file.readlines()
            lines = list(map(lambda x: x.strip(), lines))
            while [] in lines:
                lines.remove([])
        hosts = [x.split('=')[0].strip() for x in lines]
        return hosts

    def run_metis(self, graph_path, share_path):
        n_physical_hosts = len(self.get_physical_hosts(share_path))
        cmd=f"gpmetis -ptype=rb -tpwgts={str(share_path)} {str(graph_path)} {n_physical_hosts}"
        output = subprocess.check_output(cmd, shell=True)
        out = output.decode("utf-8")
        logging.debug("metis command: " + cmd)
        logging.info(out)
        return out

    def get_mapping(self, graph_path, share_path):
        gr_path = Path(graph_path)
        if gr_path.is_file():
            file_name = gr_path.name
        else:
            logging.error(f"Wrong file path in get_mapping: {graph_path}")
            raise RuntimeError()

        if Path(share_path).is_file():
            physical_hosts = self.get_physical_hosts(share_path)
        else:
            logging.error(f"Wrong file path in get_mapping: {share_path}")
            raise RuntimeError()

        mapping_file_name = file_name +".part."+ str(len(physical_hosts))
        mapping_file_path = gr_path.parent / mapping_file_name
        logging.info(f"mapping file path {mapping_file_path}")
        mapping = {host: [] for host in physical_hosts}
        with open(mapping_file_path,"r") as file:
            lines = list(map(lambda x:x.strip(), file.readlines()))

        for c, m in enumerate(lines):
            switch = c + 1
            mapping[m].append(switch)

        return mapping


    def convert_mapping(self, mapping):
        mapping_node_names = {host: [] for host in mapping.keys()}
        for host in mapping.keys():
            mapping_node_names[host] = [self.metis_node_mapping[node] for node in mapping[host]]
        return mapping_node_names

    def get_connected_hosts(self, node_name):
        nodes = []
        for node in self.virtual_network.neighbors(node_name).keys():
            if node.startswith("host"):
                nodes.append(node)

        return nodes

    def get_mapping_for_all_nodes(self, mapping_node_names):
        total_mapping={host: mapping_node_names[host] for host in mapping_node_names.keys()}
        for host in total_mapping.keys():
            for node in total_mapping[host]:
                total_mapping[host] += self.get_connected_hosts(node)

        return total_mapping

    def check_mapping_fisibility(self, mapping_node_names):
        nodes_assignement = {node: [] for node in self.virtual_network.nodes()}
        for host in mapping_node_names.keys():
            for node in mapping_node_names[host]:
                nodes_assignement[node].append(host)

        for node in nodes_assignement:
            if len(nodes_assignement[node]) == 0:
                logging.warning(f"node: {node} is not assigned")
                return False

            if len(nodes_assignement[node]) > 1:
                logging.warning(f"node: {node} is assigned to machine: {nodes_assignement[node]}")
                return False

            if len(nodes_assignement[node]) == 1:
                logging.debug(f"node: {node} is assigned to machine: {nodes_assignement[node]}")

        return True

    def build_mapping(self, metis_graph_file, metis_share_file):
        """
        the function uses the virtual_network in input and creates the metis file in the metis_graph_file path,
        then uses the share file in the metis_share_file path to run the metis partition algoritm, the result is
        saved in a file inside the  directory of metis_graph_file, after that it parse the result and return a
        dictionary having the partition.
        :param metis_graph_file: file where to put the metis graph
        :param metis_share_file: file to use for the share
        :return: dictionary of the partition
        """
        virtual_topo = self.virtual_network
        mc = MetisConverter(virtual_topo)
        mx_d = mc.convert_in_maxinet_dict()
        logging.debug("first convertion:")
        logging.debug(f"{mx_d}")
        mt_d = mc.convert_in_metis_dict(mx_d)
        logging.debug("first convertion:")
        logging.debug(f"{mt_d}")
        mt_f=mc.create_metis_file(mt_d, path=metis_graph_file)
        logging.debug("create metis file:")
        logging.debug(f"{mt_f}")
        mc.run_metis(graph_path=metis_graph_file, share_path=metis_share_file)
        mapping = mc.get_mapping(graph_path=metis_graph_file, share_path=metis_share_file)
        mapping_converted = mc.convert_mapping(mapping)
        complete_mapping = mc.get_mapping_for_all_nodes(mapping_converted)
        return complete_mapping


if __name__ == '__main__':
    virtual_topo = VirtualNetwork.create_fat_tree(k=6, density=2, req_cores=2, req_memory=4000, req_rate=500)
    converter = MetisConverter(virtual_topo)
    mapping = converter.build_mapping(metis_graph_file="metis_graphs/fattree_Metis_k_6_density_2_cores_2_mem_4000_rate_500",
                                      metis_share_file="/vagrant/bin/metis_shares/equal_6.txt")

    plotter = Plotter(virtual_topo=virtual_topo, mapping=mapping,
                      file="fattree_Metis_k_6_density_2_cores_2_mem_4000_rate_500.part.6", plotter="pyvis")
    plotter.plot()

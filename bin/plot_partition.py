import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network
import networkx as nx
from pathlib import Path

from pyvis.network import Network
from distriopt.embedding.physical import PhysicalNetwork

PLOTTERS= ["pyplot", "pyvis"]
COLORS = ['#9c27b0', '#cfd8dc', '#e8eaf6', '#ffeb3b', '#388e3c', '#4dd0e1', '#26a69a', '#3f51b5', '#b71c1c',
                 '#ffebee', '#9e9e9e', '#ff80ab', '#d500f9', '#f57c00', '#6d4c41', '#4db6ac', '#3d5afe', '#aeea00',
                 '#3f51b5', '#2196f3', '#ff5722', '#ffff00', '#4caf50', '#aed581', '#01579b', '#90a4ae', '#9fa8da',
                 '#9e9d24', '#d81b60', '#8bc34a', '#f5f5f5', '#fff3e0', '#81c784', '#bdbdbd', '#e91e63', '#6a1b9a',
                 '#ccff90', '#616161', '#ea80fc', '#00e676', '#e91e63', '#ffd600', '#303f9f', '#00b8d4', '#ffc107',
                 '#fafafa', '#43a047', '#c51162', '#e0e0e0', '#00bcd4', '#b388ff', '#ef5350', '#f1f8e9', '#64dd17',
                 '#304ffe', '#fffde7', '#827717', '#7cb342', '#c0ca33', '#84ffff', '#c62828', '#c5e1a5', '#1de9b6',
                 '#ec407a', '#f57f17', '#fdd835', '#9ccc65', '#eeff41', '#ff4081', '#ff6f00', '#ffc400', '#263238',
                 '#e040fb', '#78909c', '#3e2723', '#81d4fa', '#e57373', '#ffab40', '#2979ff', '#039be5', '#006064',
                 '#e6ee9c', '#ff5252', '#ffab91', '#ffea00', '#212121', '#efebe9', '#607d8b', '#ff6d00', '#c2185b',
                 '#ef6c00', '#b3e5fc', '#d1c4e9', '#0091ea', '#00e5ff', '#aa00ff', '#76ff03', '#ffecb3', '#ede7f6',
                 '#ffee58', '#e64a19', '#fff176', '#4527a0', '#fb8c00', '#c5cae9', '#4fc3f7', '#29b6f6', '#5c6bc0',
                 '#558b2f', '#64ffda', '#ce93d8', '#90caf9', '#0097a7', '#f06292', '#f0f4c3', '#ff9e80', '#1565c0',
                 '#ffd54f', '#d32f2f', '#26c6da', '#ffd180', '#f4511e', '#689f38', '#4a148c', '#ff8a65', '#64b5f6',
                 '#03a9f4', '#673ab7', '#00acc1', '#9c27b0', '#8c9eff', '#b2ebf2', '#fbc02d', '#d7ccc8', '#ffd740',
                 '#2e7d32', '#00bcd4', '#ffa000', '#ffe57f', '#651fff', '#e65100', '#f9a825', '#66bb6a', '#b39ddb',
                 '#f9fbe7', '#33691e', '#cddc39', '#0277bd', '#a7ffeb', '#ff8a80', '#1b5e20', '#eeeeee', '#9e9e9e',
                 '#0288d1', '#ffe082', '#283593', '#7c4dff', '#e3f2fd', '#82b1ff', '#ff6e40', '#ffb74d', '#009688',
                 '#e0f2f1', '#03a9f4', '#880e4f', '#d4e157', '#512da8', '#ff7043', '#00897b', '#f3e5f5', '#f8bbd0',
                 '#6200ea', '#7986cb', '#ffcc80', '#b2ff59', '#fff9c4', '#795548', '#ffca28', '#80deea', '#0d47a1',
                 '#004d40', '#b9f6ca', '#ffa726', '#00b0ff', '#37474f', '#ad1457', '#42a5f5', '#40c4ff', '#ff3d00',
                 '#fff59d', '#ff9800', '#00695c', '#311b92', '#f48fb1', '#f4ff81', '#cddc39', '#7b1fa2', '#80cbc4',
                 '#ff8f00', '#a1887f', '#c6ff00', '#ffc107', '#f44336', '#ff9100', '#00bfa5', '#b2dfdb', '#009688',
                 '#795548', '#5e35b1', '#e8f5e9', '#fbe9e7', '#ffb300', '#bcaaa4', '#757575', '#b0bec5', '#2962ff',
                 '#8d6e63', '#18ffff', '#ffab00', '#ff9800', '#dcedc8', '#80d8ff', '#ffff8d', '#ff5722', '#e53935',
                 '#424242', '#8e24aa', '#1976d2', '#8bc34a', '#00796b', '#fff8e1', '#f44336', '#69f0ae', '#c8e6c9',
                 '#fce4ec', '#afb42b', '#ffcdd2', '#536dfe', '#bbdefb', '#00c853', '#d50000', '#5d4037', '#eceff1',
                 '#9575cd', '#2196f3', '#e0f7fa', '#ff1744', '#1a237e', '#ffccbc', '#ba68c8', '#a5d6a7', '#bf360c',
                 '#4caf50', '#00838f', '#448aff', '#dce775', '#ab47bc', '#f50057', '#e1bee7', '#673ab7', '#7e57c2',
                 '#e1f5fe', '#d84315', '#546e7a', '#ffe0b2', '#4e342e', '#1e88e5', '#dd2c00', '#455a64', '#ef9a9a',
                 '#607d8b', '#3949ab', '#ffeb3b']

class Plotter(object):
    def __init__(self,virtual_topo, mapping, file, plotter="pyplot"):
        if not plotter in PLOTTERS:
            raise Exception(f"Plotter error: {plotter}")

        self.virtual_topo = virtual_topo
        self.plotter = plotter
        self.mapping = mapping
        self.file = file
        self.path = Path(file).parent
        self.file_name = Path(file).name
        self.image_path = self.path /"metis_graphs"/ "img" / self.file_name

    def plot(self):
        if self.plotter == "pyplot":
            self.__pyplot()
        if self.plotter == "pyvis":
            self.__pyvis()

    def __pyplot(self):
        edges = self.virtual_topo.edges()
        hosts = self.mapping.keys()
        G = nx.Graph()
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, k=2)
        labels = {x: x for x in self.virtual_topo.nodes()}
        petches = []
        for c, host in enumerate(hosts):
            nodelist= self.mapping[host]
            nodecolor = COLORS[c]
            petches.append(mpatches.Patch(color=nodecolor, label=host))
            nx.draw_networkx_nodes(G, pos, node_color=nodecolor, nodelist=nodelist, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.axis('off')
        plt.legend(handles=petches)
        plt.savefig(str(self.image_path)+".png")

    def __pyvis(self):
        edges = self.virtual_topo.edges()
        hosts = self.mapping.keys()

        vis = Network(height="750px", width="100%", font_color="black")
        for c, host in enumerate(hosts):
            vis.add_nodes(self.mapping[host], label=self.mapping[host],
                      color=[COLORS[c] for i in self.mapping[host]])

        vis.add_edges(edges)

        vis.show_buttons(filter_=['physics'])
        vis.show(name=str(self.image_path)+".html")

    @staticmethod
    def plot_physical_network(physical_network, image_path):
        """
        Takes in input an obj PhysicalNetwork from distriopt.embedding.physical
        :param physica_network:
        :return:
        """
        edges = physical_network.edges()
        weighted_edges = []
        for u, v in edges:
            iface=physical_network.interfaces_ids(u, v)
            iface = iface[0]
            rate = iface["rate"]
            weighted_edges.append((u,v,rate))

        G = nx.Graph()
        print(weighted_edges)
        G.add_weighted_edges_from(weighted_edges)

        pos = nx.spring_layout(G,k=5)
        nx.draw_networkx(G, pos,arrows=False,label=True)
        nx.draw_networkx_edge_labels(G, pos,edge_labels={(u, v): w for u, v, w in weighted_edges},font_size=5)
        plt.axis('off')
        plt.savefig(str(image_path) + ".png")


if __name__ == '__main__':

    physical = PhysicalNetwork.from_files("/vagrant/bin/physical_instances/random5")
    print(physical.edges())
    Plotter.plot_physical_network(physical,"/vagrant/bin/physical_instances/random5")

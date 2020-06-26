import json
import pickle

from experiment import Experiment
from plot_partition import Plotter
from distriopt import VirtualNetwork
from pathlib import Path
from distriopt.embedding.physical import PhysicalNetwork
import networkx as nx
from metis_converter import MetisConverter
from distriopt.embedding.algorithms import (
    EmbedBalanced,
    EmbedILP,
    EmbedPartition,
    EmbedGreedy,
)

from mininet.topolib import TreeTopo
from mininet.topo import Topo
import random

NotSolved = 0
Solved = 1
Infeasible = -1

class ___(object):
    def __init__(self):
        self.node_mapping = {}

class SwitchBinPlacer( object ):
    """Place switches (and controllers) into evenly-sized bins,
       and attempt to co-locate hosts and switches"""

    def __init__( self, virtual_topo, physical_topo):
        # Easy lookup for servers and node sets
        self.servers=list(physical_topo.compute_nodes)
        self.controllers=[]
        self.links=self.getlinks(virtual_topo)
        self.servdict = dict( enumerate( physical_topo.compute_nodes ) )
        self.hosts = self.gethosts(virtual_topo)
        self.hset = frozenset( self.hosts )
        self.switches = self.getswitches(virtual_topo)
        self.sset = frozenset( self.switches )
        self.cset = frozenset( self.controllers )
        # Server and switch placement indices
        self.placement = self.calculatePlacement()

        self.solution = ___()

    def solve(self):
        self.solution.node_mapping=self.placement
        return 1,1

    def getlinks(self, virtual_topo):
        return virtual_topo.edges()

    def gethosts(self, virtual_topo):
        nodes=[]
        for n in virtual_topo.nodes():
            if n.startswith("host"):
                nodes.append(n)
        return nodes

    def getswitches(self, virtual_topo):
        nodes = []
        for n in virtual_topo.nodes():
            if not n.startswith("host"):
                nodes.append(n)
        return nodes

    @staticmethod
    def bin( nodes, servers ):
        "Distribute nodes evenly over servers"
        # Calculate base bin size
        nlen = len( nodes )
        slen = len( servers )
        # Basic bin size
        quotient = int( nlen / slen )
        binsizes = { server: quotient for server in servers }
        # Distribute remainder
        remainder = nlen % slen
        for server in servers[ 0 : remainder ]:
            binsizes[ server ] += 1
        # Create binsize[ server ] tickets for each server
        tickets = sum( [ binsizes[ server ] * [ server ]
                         for server in servers ], [] )
        # And assign one ticket to each node
        return { node: ticket for node, ticket in zip( nodes, tickets ) }

    def calculatePlacement( self ):
        "Pre-calculate node placement"
        placement = {}
        # Create host-switch connectivity map,
        # associating host with last switch that it's
        # connected to
        switchFor = {}
        for src, dst in self.links:
            if src in self.hset and dst in self.sset:
                switchFor[ src ] = dst
            if dst in self.hset and src in self.sset:
                switchFor[ dst ] = src
        # Place switches
        placement = self.bin( self.switches, self.servers )
        # Place controllers and merge into placement dict
        placement.update( self.bin( self.controllers, self.servers ) )
        # Co-locate hosts with their switches
        for h in self.hosts:
            if h in placement:
                # Host is already placed - leave it there
                continue
            if h in switchFor:
                placement[ h ] = placement[ switchFor[ h ] ]
            else:
                raise Exception(
                        "SwitchBinPlacer: cannot place isolated host " + h )
        return placement

    def place( self, node ):
        """Simple placement algorithm:
           place switches into evenly sized bins,
           and place hosts near their switches"""
        return self.placement[ node ]



class RandomPlacer(object):

    def __init__(self, virtual_topo, physical_topo):
        self.virtual_topo=virtual_topo
        self.physical_topo=list(physical_topo.compute_nodes)
        self.solution = ___()

    def solve(self):
        placement={}
        for v in self.virtual_topo.nodes():
            placement[v] = random.choice(self.physical_topo)
        self.solution.node_mapping = placement

        return 1,1

class RoundRobin(object):

    def __init__(self, virtual_topo, physical_topo):
        self.virtual_topo=virtual_topo
        self.physical_topo=list(physical_topo.compute_nodes)
        self.solution = ___()

    def solve(self):
        placement={}
        c = 0
        for v in self.virtual_topo.nodes():
            placement[v] = self.physical_topo[c % len(self.physical_topo)]
            c+=1
        self.solution.node_mapping = placement

        return 1, 1

def save_logs(pickle_file,file_name,cpu_over,memory_over,link_over,nodes):
    infile = open(pickle_file,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    new_dict[file_name]={"cpu":cpu_over,"memory":memory_over,"link":link_over, "nodes":nodes}
    outfile = open(pickle_file, 'wb')
    pickle.dump(new_dict, outfile)
    outfile.close()

def MetisSolution(filename,physical_file,metis_share_file="/vagrant/bin/metis_shares/equal_14.txt", virtual_topology=None, pickle_file="log.pkl"):
    #filename = "fat_tree_k8_d2"
    #physical_file = "/vagrant/bin/physical_instances/grele"
    physical = PhysicalNetwork.from_files(physical_file)
    #virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores, req_memory=req_memory, req_rate=req_rate)
    virtual_topo=virtual_topology
    converter = MetisConverter(virtual_topo)
    mapping = converter.build_mapping(metis_graph_file=f"/vagrant/bin/metis_graphs/{filename}",
                                      metis_share_file=metis_share_file)

    ##SORTING TO GIVE TO THE BIG NUMBERS THE BIGGEST MACHINES
    compute_nodes=sorted(list(physical.compute_nodes))
    sorted_keys=sorted(mapping.keys(), key=lambda x: int(x), reverse=True)
    print(sorted_keys)
    print(compute_nodes)
    physical_names_mapping = {phy_name: metis_name for phy_name, metis_name in
                              zip(compute_nodes, sorted_keys)}

    metis_name_mapping = {physical_names_mapping[x]: x for x in physical_names_mapping.keys()}

    mapping_with_pyhisical_names = {metis_name_mapping[node]: mapping[node] for node in mapping.keys()}
    print(mapping)

    exp = Experiment(physical_topo=physical, solution_dict=mapping_with_pyhisical_names, virtual_topo=virtual_topo)
    cpu_over = exp.cpu_overcommitment()
    memory_over = exp.memory_overcommitment()
    link_over = exp.total_links_overcommitment()
    node_used = exp.node_used()
    print(node_used)
    print(max(cpu_over.values()), cpu_over)
    print(max(memory_over.values()), memory_over)
    print(max(link_over.values()), link_over)
    save_logs(pickle_file=pickle_file,file_name=filename,cpu_over=cpu_over,memory_over=memory_over,link_over=link_over,nodes=node_used)

    #plotter = Plotter(virtual_topo=virtual_topo, mapping=mapping_with_pyhisical_names, file=filename, plotter="pyplot")
    #plotter.plot()
    #plotter1 = Plotter(virtual_topo=virtual_topo, mapping=mapping_with_pyhisical_names, file=filename, plotter="pyvis")
    #plotter1.plot()

def HeuristicSolution(filename,physical_file,heuristic=EmbedGreedy, virtual_topology=None,pickle_file="log.pkl"):
    physical = PhysicalNetwork.from_files(physical_file)
    #virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores, req_memory=req_memory,req_rate=req_rate)
    virtual_topo= virtual_topology
    prob = heuristic(virtual_topo, physical)
    time_solution, status = prob.solve()
    print(time_solution, status)
    if status == -1:
        save_logs(pickle_file=pickle_file, file_name=filename, cpu_over=None, memory_over=None, link_over=None, nodes=None)
        return

    solution = prob.solution.node_mapping
    reverse_mapping = {x: [] for x in set(solution.values())}
    for vnode in solution:
        reverse_mapping[solution[vnode]].append(vnode)
    print(reverse_mapping)


    #filename1 = "andrea_algo"
    exp1 = Experiment(physical_topo=physical, solution_dict=reverse_mapping, virtual_topo=virtual_topo)
    cpu_over = exp1.cpu_overcommitment()
    memory_over = exp1.memory_overcommitment()
    link_over = exp1.total_links_overcommitment()
    node_used = exp1.node_used()
    print(node_used)
    print(max(cpu_over.values()), cpu_over)
    print(max(memory_over.values()), memory_over)
    print(max(link_over.values()), link_over)
    save_logs(pickle_file=pickle_file,file_name=filename,cpu_over=cpu_over,memory_over=memory_over,link_over=link_over,nodes=node_used)


    #plotter = Plotter(virtual_topo=virtual_topo, mapping=reverse_mapping, file=filename, plotter="pyplot")
    #plotter.plot()
    #plotter1 = Plotter(virtual_topo=virtual_topo, mapping=reverse_mapping, file=filename, plotter="pyvis")
    #plotter1.plot()

def main(physical_topology_name="Lyon", pickle_file= "lyon_fattree.pkl",
         physical_file = "/vagrant/bin/physical_instances/lyon",
        metis_share_file="/vagrant/bin/metis_shares/lyon.txt"
         ):
    heuristics={
        "EmbedBalanced": EmbedBalanced,
        #"EmbedILP": EmbedILP,
        "EmbedPartition": EmbedPartition,
        "EmbedGreedy": EmbedGreedy,
        "RoundRobin": RoundRobin,
        "SwitchBinPlacer": SwitchBinPlacer,
        "RandomPlacer": RandomPlacer
    }

    outfile = open(pickle_file, 'wb')
    pickle.dump(dict(), outfile)
    outfile.close()
    for k in [2, 4, 6, 8, 10]:
        for density in [int(k/2)]:
            for req_cores in [1, 2, 4, 6, 8, 10]:
                for req_memory in [100, 1000, 2000, 4000, 8000, 16000]:
                    for req_rate in [1, 10, 50, 100, 200, 500, 1000]:
                        print(k,density,req_cores,req_memory,req_rate)
                        metis_filename=f"{physical_topology_name}_Metis_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"
                        virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores, req_switch_cores=req_cores, req_switch_memory=req_memory, req_memory=req_memory, req_rate=req_rate)
                        MetisSolution(filename=metis_filename,
                                      physical_file=physical_file,
                                      metis_share_file=metis_share_file,
                                      virtual_topology=virtual_topo,
                                      pickle_file=pickle_file
                                      )

                        for h in heuristics.keys():
                            heuristic = heuristics[h]
                            heuristic_filename = f"{physical_topology_name}_{h}_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"
                            virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores, req_switch_cores=req_cores, req_switch_memory=req_memory, req_memory=req_memory, req_rate=req_rate)
                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file,
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)








def main1():
    heuristics = {
        "EmbedBalanced": EmbedBalanced,
        # "EmbedILP": EmbedILP,
        "EmbedPartition": EmbedPartition,
        "EmbedGreedy": EmbedGreedy
    }
    heuristics_new = {
        "RoundRobin": RoundRobin,
        "SwitchBinPlacer": SwitchBinPlacer,
        "RandomPlacer": RandomPlacer
    }
    pickle_file = "gros_forcing_hosts_fattree.pkl"
    outfile = open(pickle_file, 'wb')
    pickle.dump(dict(), outfile)
    outfile.close()
    for k in [2, 4, 6, 8, 10]:
        for density in [int(k/2)]:
            for req_cores in [1, 2, 4, 6, 8, 10]:
                for req_memory in [100, 1000, 2000, 4000, 8000, 16000]:
                    for req_rate in [1, 10, 50, 100, 200, 500, 1000]:
                        physical_file = "/vagrant/bin/physical_instances/gros20"
                        for h in ["EmbedGreedy"]:
                            heuristic = heuristics[h]
                            heuristic_filename = f"fattree_{h}_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"
                            virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores,
                                                                          req_switch_cores=req_cores, req_switch_memory=req_memory,
                                                                          req_memory=req_memory, req_rate=req_rate)
                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file,
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)



                        infile = open(pickle_file, 'rb')
                        new_dict = pickle.load(infile)
                        infile.close()
                        sol=new_dict[f"fattree_EmbedGreedy_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"]
                        print(sol["nodes"])
                        if sol["nodes"]== None:
                            continue
                        else:
                            node_used_by_emb_greedy=len(sol["nodes"])

                        if node_used_by_emb_greedy == 1:
                            continue

                        physical_file = "/vagrant/bin/physical_instances/gros"
                        metis_filename = f"fattree_Metis_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"

                        MetisSolution(filename=metis_filename,
                                      physical_file=physical_file+str(node_used_by_emb_greedy),
                                      metis_share_file=f"/vagrant/bin/metis_shares/equal_{node_used_by_emb_greedy}.txt",
                                      virtual_topology=virtual_topo,
                                      pickle_file=pickle_file
                                      )

                        for h in heuristics_new.keys():
                            heuristic = heuristics_new[h]
                            heuristic_filename = f"fattree_{h}_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"

                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file+str(node_used_by_emb_greedy),
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)




def main1_vrandom():
    heuristics = {
        "EmbedBalanced": EmbedBalanced,
        # "EmbedILP": EmbedILP,
        "EmbedPartition": EmbedPartition,
        "EmbedGreedy": EmbedGreedy
    }
    heuristics_new = {
        "RoundRobin": RoundRobin,
        "SwitchBinPlacer": SwitchBinPlacer,
        "RandomPlacer": RandomPlacer
    }
    physical_topology_name="gros"
    pickle_file = "gros_forcing_hosts_random.pkl"
    outfile = open(pickle_file, 'wb')
    pickle.dump(dict(), outfile)
    outfile.close()
    for n_nodes in range(10,51,5):
        for seed in range(100):
            for d in [0.1,0.2,0.3,0.4,0.5]:
                        physical_file = "/vagrant/bin/physical_instances/gros20"

                        for h in ["EmbedGreedy"]:
                            heuristic = heuristics[h]
                            heuristic_filename = f"{physical_topology_name}_{h}_n_nodes_{n_nodes}_density_{d}_seed_{seed}"
                            virtual_topo = VirtualNetwork.create_random_nw(n_nodes=n_nodes, p=d,
                                                                           req_cores=[1, 2, 3, 4, 5, 6, 7, 8],
                                                                           req_memory=[1000, 2000, 4000, 6000, 8000,
                                                                                       16000],
                                                                           req_rate=[1, 10, 50, 100, 200], seed=seed)
                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file,
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)



                        infile = open(pickle_file, 'rb')
                        new_dict = pickle.load(infile)
                        infile.close()
                        sol=new_dict[f"{physical_topology_name}_{h}_n_nodes_{n_nodes}_density_{d}_seed_{seed}"]
                        print(sol["nodes"])
                        if sol["nodes"]== None:
                            continue
                        else:
                            node_used_by_emb_greedy=len(sol["nodes"])

                        if node_used_by_emb_greedy == 1:
                            continue

                        physical_file = "/vagrant/bin/physical_instances/gros"
                        metis_filename = f"{physical_topology_name}_Metis_n_nodes_{n_nodes}_density_{d}_seed_{seed}"

                        MetisSolution(filename=metis_filename,
                                      physical_file=physical_file+str(node_used_by_emb_greedy),
                                      metis_share_file=f"/vagrant/bin/metis_shares/equal_{node_used_by_emb_greedy}.txt",
                                      virtual_topology=virtual_topo,
                                      pickle_file=pickle_file
                                      )

                        for h in heuristics_new.keys():
                            heuristic = heuristics_new[h]
                            heuristic_filename = f"{physical_topology_name}_{h}_n_nodes_{n_nodes}_density_{d}_seed_{seed}"

                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file+str(node_used_by_emb_greedy),
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)

def mainRandom(physical_topology_name="Random3",pickle_file= "Random3_fattree.pkl",
         physical_file = "/vagrant/bin/physical_instances/random3",
        metis_share_file="/vagrant/bin/metis_shares/random_3.txt"):
    heuristics={
        "EmbedBalanced": EmbedBalanced,
        #"EmbedILP": EmbedILP,
        "EmbedPartition": EmbedPartition,
        "EmbedGreedy": EmbedGreedy,
        "RoundRobin": RoundRobin,
        "SwitchBinPlacer": SwitchBinPlacer,
        "RandomPlacer": RandomPlacer
    }
    outfile = open(pickle_file, 'wb')
    pickle.dump(dict(), outfile)
    outfile.close()
    for n_nodes in range(10,51,5):
        for seed in range(100):
            for d in [0.1,0.2,0.3,0.4,0.5]:
                print(n_nodes,seed,d)
                print("metis")
                virtual_topo = VirtualNetwork.create_random_nw(n_nodes=n_nodes, p=d, req_cores=[1, 2, 3, 4, 5, 6, 7, 8], req_memory=[1000, 2000, 4000, 6000, 8000, 16000], req_rate=[1, 10, 50, 100, 200], seed=seed)
                metis_filename=f"{physical_topology_name}_Metis_n_nodes_{n_nodes}_density_{d}_seed_{seed}"
                MetisSolution(filename=metis_filename,
                              physical_file=physical_file,
                              metis_share_file=metis_share_file,
                              virtual_topology=virtual_topo,
                              pickle_file=pickle_file
                              )

                for h in heuristics.keys():
                    heuristic = heuristics[h]
                    print(h)
                    heuristic_filename=f"{physical_topology_name}_{h}_n_nodes_{n_nodes}_density_{d}_seed_{seed}"
                    HeuristicSolution(filename=heuristic_filename,
                                      physical_file=physical_file,
                                      heuristic=heuristic,
                                      virtual_topology=virtual_topo,
                                      pickle_file=pickle_file)




def main5(physical_topology_name="lyon",pickle_file= "Lyon_fattree.pkl",
         physical_file = "/vagrant/bin/physical_instances/lyon",
        metis_share_file="/vagrant/bin/metis_shares/lyon.txt"
         ):
    heuristics={
        "EmbedBalanced": EmbedBalanced,
        #"EmbedILP": EmbedILP,
        "EmbedPartition": EmbedPartition,
        "EmbedGreedy": EmbedGreedy,
        "RoundRobin": RoundRobin,
        "SwitchBinPlacer": SwitchBinPlacer,
        "RandomPlacer": RandomPlacer
    }

    outfile = open(pickle_file, 'wb')
    pickle.dump(dict(), outfile)
    outfile.close()
    for k in [4]:
        for density in [int(k/2)]:
            for req_cores in [4]:
                for req_memory in [65000]:
                    for req_rate in [500]:
                        print("Metis")
                        metis_filename=f"{physical_topology_name}_Metis_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"
                        virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density, req_cores=req_cores,
                                                                      req_switch_cores=1, req_switch_memory =1000,
                                                                      req_memory=req_memory, req_rate=req_rate)
                        MetisSolution(filename=metis_filename,
                                      physical_file=physical_file,
                                      metis_share_file=metis_share_file,
                                      virtual_topology=virtual_topo,
                                      pickle_file=pickle_file
                                      )


                        for h in heuristics.keys():
                            print(h)
                            heuristic = heuristics[h]
                            heuristic_filename = f"{physical_topology_name}_{h}_k_{k}_density_{density}_cores_{req_cores}_mem_{req_memory}_rate_{req_rate}"
                            virtual_topo = VirtualNetwork.create_fat_tree(k=k, density=density,
                                                                          req_cores=req_cores, req_switch_cores =1,
                                                                          req_switch_memory=2000, req_memory=req_memory,req_rate=req_rate)
                            HeuristicSolution(filename=heuristic_filename,
                                              physical_file=physical_file,
                                              heuristic=heuristic,
                                              virtual_topology=virtual_topo,
                                              pickle_file=pickle_file)



if __name__ == '__main__':
    
    main(physical_topology_name="Rennes", pickle_file= "Rennes_fattree.pkl",
         physical_file = "/vagrant/bin/physical_instances/rennes",
        metis_share_file="/vagrant/bin/metis_shares/rennes.txt")
    mainRandom(physical_topology_name="Rennes", pickle_file= "Rennes_random.pkl",
               physical_file="/vagrant/bin/physical_instances/rennes",
               metis_share_file="/vagrant/bin/metis_shares/rennes.txt")
    
    
    main1()
    main1_vrandom()
    
    print("\n\n####################################  LYON #############################################")
    """
    main5(physical_topology_name="lyon",pickle_file= "_.pkl",
         physical_file = "/vagrant/bin/physical_instances/lyon",
        metis_share_file="/vagrant/bin/metis_shares/lyon.txt")

    print("\n\n#################################### LYON 1 #############################################")

    main5(physical_topology_name="lyon_1",pickle_file= "_.pkl",
         physical_file = "/vagrant/bin/physical_instances/lyon_1",
        metis_share_file="/vagrant/bin/metis_shares/lyon_1.txt")


    print("\n\n#################################### GRISOU 10 #############################################")

    main5(physical_topology_name="grisou10", pickle_file="_.pkl",
          physical_file="/vagrant/bin/physical_instances/grisou10",
          metis_share_file="/vagrant/bin/metis_shares/equal_10.txt")
    print("\n\n#################################### GRISOU 20 #############################################")

    main5(physical_topology_name="grisou20", pickle_file="_.pkl",
          physical_file="/vagrant/bin/physical_instances/grisou20",
          metis_share_file="/vagrant/bin/metis_shares/equal_20.txt")


    """
    print("\n\n#################################### GROS 20 #############################################")


    main5(physical_topology_name="gros", pickle_file="_.pkl",
          physical_file="/vagrant/bin/physical_instances/gros",
          metis_share_file="/vagrant/bin/metis_shares/equal_20.txt")
          
    
    main(physical_topology_name="Gros", pickle_file= "Gros_fattree.pkl",
         physical_file = "/vagrant/bin/physical_instances/gros",
        metis_share_file="/vagrant/bin/metis_shares/equal20.txt")
    mainRandom(physical_topology_name="Gros", pickle_file= "Gros_random.pkl",
               physical_file="/vagrant/bin/physical_instances/gros",
               metis_share_file="/vagrant/bin/metis_shares/equal20.txt.txt")
    

import csv
import subprocess
from collections import defaultdict
import argparse

partition_csv = "<your_path>/partitions.csv"
result = subprocess.run(['sinfo'], stdout=subprocess.PIPE, check=True)
output = result.stdout.decode('utf-8')
lines = output.split('\n')
partition_list = {} # {"<status>" : {"<partition>" : ["<node>", "<node>", ...]}}
max_tres = 48
PRICES = {
    "a100" : 1.89,
    "4090" : 0.74,
    "a6000" : 0.79,
    "3090" : 0.44
}

parser = argparse.ArgumentParser(description='Get idle partitions and recommend commands')
parser.add_argument('--sbatch', action='store_true', help='Print sbatch instead of srun')
parser.add_argument('--partition', type=str, help='Partition name to get info', default="")
parser.add_argument("--max_tres", type=int, help="Maximum TRES", default=48)
args = parser.parse_args()

max_tres = args.max_tres

if args.sbatch:
    def printif(something):
        pass
else:
    def printif(something):
        print(something)



class NodeInfoParser:
    def __init__(self, node_name:str):
        self.node_name = node_name
        self.node_info = {}
        self.output = ""
        self.device_name = ""
        self.total_device_count = 0
        self.partitions = ""
        self.available_device_count = 0
    
    def get_command(self):
        return f"scontrol show node {self.node_name}".split()

    def get_cpu_available_count(self):
        stdout = self.output
        cpu_alloc = stdout.split("CPUAlloc=")[1].split()[0]
        cpu_tot = stdout.split("CPUTot=")[1].split()[0]
        return int(cpu_tot) - int(cpu_alloc)
        
    def get_node_info(self):
        result = subprocess.run(self.get_command(), stdout=subprocess.PIPE, check=True)
        self.output = result.stdout.decode('utf-8')
        # find Gres=<gres>
        gres_start = self.output.index('Gres=') + len('Gres=')
        gres_end = self.output.index('NodeAddr=') - 2
        gres = self.output[gres_start:gres_end]
        # split by ":"
        gres_found = False
        if gres != "(null)": # no Gres
            gres_found = True
        #printif("gres", gres)
        self.node_info['gres'] = {}
        self.device_name = gres.split(":")[0]
        self.total_device_count = gres.split(':')[-1]
        if self.total_device_count == "0":
            printif(f"No info from {gres}")
        # if not numeric self.total_device_count, check CfgTres=cpu=52,...
        if not self.total_device_count.isnumeric():
            cfg_start = self.output.index('CfgTRES=') + len('CfgTRES=')
            cfg_end = self.output.index('\n', cfg_start)
            cfg_tres = self.output[cfg_start:cfg_end]
            #printif("cfg_tres", cfg_tres)
            # if gres/ in cfg_tres, use it
            if 'gres/' in cfg_tres:
                #printif(f"cfg tres for {self.device_name} : {cfg_tres}")
                cfg_tres = cfg_tres.split('gres/')[-1]
            cfg_tres = cfg_tres.split(',')[0]
            self.node_info['gres'][self.device_name] = cfg_tres.split('=')[-1]
            assert self.node_info['gres'][self.device_name], "no info?"
            self.gres_name = self.device_name
            self.available_device_count = self.total_device_count = self.node_info['gres'][self.device_name]
            assert self.available_device_count.isnumeric()
        # get AllocTRES
        alloc_start = self.output.index('AllocTRES=') + len('AllocTRES=')
        # until next \n
        alloc_end = self.output.index('\n', alloc_start)
        alloc_tres = self.output[alloc_start:alloc_end]
        if not alloc_tres or alloc_tres.isspace():
            # is idle
            # parse CfgTRES instead, gres/gpu=6\n
            cfg_start = self.output.index('CfgTRES=') + len('CfgTRES=')
            cfg_end = self.output.index('\n', cfg_start)
            cfg_tres = self.output[cfg_start:cfg_end]
            self.node_info['gres'][self.device_name] = cfg_tres.split('=')[-1]
            self.gres_name = self.device_name
            self.available_device_count = int(self.total_device_count)
        else:
            # split by ","
            alloc_tres_list = alloc_tres.split(',')
            any_gres_found = False
            for tres in alloc_tres_list:
                #printif("tres", tres)
                if tres.startswith("gres/"):
                    tres = tres[len("gres/"):]
                    device_name = tres.split("=")[0]
                    count = tres.split('=')[-1]
                    self.node_info['gres'][device_name] = count
                    self.gres_name = device_name
                    self.available_device_count = int(self.total_device_count) - int(count)
                    any_gres_found = True
            if not any_gres_found:
                # if a=count format, just use it
                if '=' in alloc_tres:
                    self.node_info['gres'][self.device_name] = alloc_tres.split('=')[-1]
                    self.gres_name = self.device_name
                    self.available_device_count = int(self.total_device_count)
                else:
                    raise Exception(f"No Gres found in {self.node_name}! {alloc_tres_list}")
        # get Partitions
        partition_start = self.output.index('Partitions=') + len('Partitions=')
        partition_end = self.output.index('\n', partition_start)
        partitions = self.output[partition_start:partition_end]
        self.cpu_count_per_gpu = self.get_cpu_available_count() // max(self.available_device_count,1)
        self.partitions = partitions
        return self.node_info
    
    def get_recommended_command(self, qos_name:str):
        self.get_node_info()
        if "null" in self.gres_name:
            return f"[{self.node_name}] srun --partition={self.partitions} --time=72:0:0 --nodes=1 --qos={qos_name.strip()} --pty bash -i"
        return f"[{self.node_name}] srun --partition={self.partitions} --time=72:0:0 --nodes=1 --cpus-per-gpu={self.cpu_count_per_gpu} --qos={qos_name.strip()} --gres={self.gres_name}:{self.available_device_count} --pty bash -i"
    def get_gpus_and_cpus_count(self):
        return self.partitions, int(self.available_device_count), int(self.cpu_count_per_gpu)
class StringNodeParser:
    # node01 -> node01
    # node[01,03-05,07] -> node01,node03,node04,node05,node07
    # anode[01-03],bnode[01-03] -> anode01,anode02,anode03,bnode01,bnode02,bnode03
    # anode[01,03-05,07],bnode[01,03-05,07] -> anode01,anode03,anode04,anode05,anode07,bnode01,bnode03,bnode04,bnode05,bnode07
    def __init__(self, node_string):
        self.node_string = node_string
        self.node_list = []
        self.parse()
    def parse(self):
        # get "," which is not inside []
        comma_list = []
        inside_bracket = False
        for i, c in enumerate(self.node_string):
            if c == '[':
                inside_bracket = True
            elif c == ']':
                inside_bracket = False
            elif c == ',' and not inside_bracket:
                comma_list.append(i)
        # split by ","
        start = 0
        for i in comma_list:
            self.parse_node(self.node_string[start:i]) # anode[01-03] for example
            start = i + 1
        self.parse_node(self.node_string[start:])
    def parse_node(self, node_string):
        # if no bracket, just append
        if '[' not in node_string:
            self.node_list.append(node_string)
            return
        # parse bracket
        bracket_start = node_string.index('[')
        bracket_end = node_string.index(']')
        prefix = node_string[:bracket_start]
        # no suffix
        # if singleton inside bracket (no comma), just append
        if ',' not in node_string[bracket_start:bracket_end] and "-" not in node_string[bracket_start:bracket_end]:
            self.node_list.append(prefix + node_string[bracket_start:bracket_end + 1])
            return
        # we can safely assume that there is no nested bracket so just split by comma
        ranges = node_string[bracket_start + 1:bracket_end].split(',')
        for r in ranges:
            if '-' in r:
                start, end = r.split('-')
                for i in range(int(start), int(end) + 1):
                    self.node_list.append(prefix + str(i).zfill(len(start)))
            else:
                self.node_list.append(prefix + r)
    def get_node_list(self):
        return self.node_list

for line in lines:
    # PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    if line.startswith('PARTITION'):
        continue
    if line == '':
        continue
    partiition, avail, timelimit, nodes, state, nodelist = line.split()
    # parse nodelist, nodename[01,03-05,07],... -> nodename01,nodename03,nodename04,nodename05,nodename07
    # check singleton node
    node_parser = StringNodeParser(nodelist)
    node_list = node_parser.get_node_list()
    #printif(node_list)
    if state not in partition_list:
        partition_list[state] = {}
    partition_list[state][partiition] = node_list
    
# print partition_list
# print list of idle partitions
printif(f"Idle partitions: {list(partition_list.get('idle', {}).keys())} with nodes {list( partition_list.get('idle', {}).values())}")
# csv contains Partition Name,Allowed QoS Names
# read csv
idle_commands = []
mix_commands = []
all_infos = []
with open(partition_csv, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    empty_gpus = {}
    for row in csvreader:
        partition_name, qos_list, _ = row
        if partition_name in ["test", "maintenance"]:
            continue
        qos_list = qos_list.split('|')
        if partition_name in partition_list.get('idle', {}):
            #printif(f"Partition {partition_name} is idle with nodes {partition_list['idle'][partition_name]} and allowed QoS {qos_list}")
            # recommend srun --partition=suma_a100 --time=2:0 --nodes=1 --qos a100_qos --gres=gpu:1 --pty bash -i like command
            #idle_commands.append(f"srun --partition={partition_name} --time=2:0 --nodes=1 --qos={qos_list[-1]} --gres=gpu:1 --pty bash -i")
            for i in partition_list['idle'][partition_name]:
                parser = NodeInfoParser(i)
                idle_commands.append(parser.get_recommended_command(qos_list[-1]))
                all_infos.append(parser.get_gpus_and_cpus_count())
                if partition_name not in empty_gpus:
                    empty_gpus[partition_name] = 0
                empty_gpus[partition_name] += parser.available_device_count
            #printif(NodeInfoParser(partition_list['idle'][partition_name][0]).get_recommended_command(qos_list[-1]))
        if partition_name in partition_list["mix"]:
            # get detailed info
            for i in partition_list["mix"][partition_name]:
                parser = NodeInfoParser(i)
                mix_commands.append(parser.get_recommended_command(qos_list[-1]))
                all_infos.append(parser.get_gpus_and_cpus_count())
                if partition_name not in empty_gpus:
                    empty_gpus[partition_name] = 0
                empty_gpus[partition_name] += parser.available_device_count
            #printif(NodeInfoParser(partition_list["mix"][partition_name][0]).get_recommended_command(qos_list[-1]))
        # mix, get more detailed info
from collections import defaultdict

# Function to compute the maximum node count * GPUs for a given minimal CPU count
def max_nodes_x_gpus(data, cpu_min):
    max_product = {}

    # Filter out entries with a CPU count of zero and entries below the specified minimal CPU count
    filtered_data = [entry for entry in data if entry[2] != 0 and entry[2] >= cpu_min]

    # Compute the maximum product of nodes and GPUs for each partition
    for name, gpus, cpu_count in filtered_data:
        matching_entries = [x for x in filtered_data if x[0] == name and x[1] >= gpus and x[2] >= cpu_count]
        match_count = len(matching_entries)
        product_num = gpus * match_count
        if product_num > max_tres:
            while product_num > max_tres:
                product_num -= gpus
                match_count -= 1
        if match_count == 0 or product_num == 0:
            continue
        if name not in max_product:
            max_product[name] = (product_num, gpus, match_count, cpu_count)
        elif product_num > max_product[name][0]: # smaller nodes are better
            max_product[name] = (product_num, gpus, match_count, cpu_count)
        elif product_num == max_product[name][0] and cpu_count > max_product[name][3] and max_product[name][2] >= match_count:
            max_product[name] = (product_num, gpus, match_count, cpu_count)
    return max_product

# Example usage
cpu_min = 5
max_product_infos = max_nodes_x_gpus(all_infos, cpu_min)

# Print the results for each partition name
for name in sorted(max_product_infos):
    # print name, total gpus, gpus per node, nodes, cpus per gpu
    printif(f"{name.strip()}: total gpus {max_product_infos[name][0]}, gpus per node {max_product_infos[name][1]}, nodes {max_product_infos[name][2]}, cpus per gpu {max_product_infos[name][3]}")
    if args.partition in name.strip() and args.sbatch:
        # format
        # SBATCH --partition=name
        # SBATCH --time=72:0:0
        # SBATCH --nodes={max_product_infos[name][2]}
        # SBATCH --cpus-per-gpu={max_product_infos[name][3]}
        # SBATCH --gres={gres_name}:{max_product_infos[name][1]}
        print(f"#SBATCH --nodes={max_product_infos[name][2]}")
        print(f"#SBATCH --cpus-per-gpu={max_product_infos[name][3]}")
        print(f"#SBATCH --gres=gpu:{max_product_infos[name][1]}")
printif("Idle commands:")
for c in idle_commands:
    if "gpu:0" in c:
        continue
    printif(c)
printif("Mix commands:")
for c in mix_commands:
    if "gpu:0" in c:
        continue
    printif(c)

# print empty gpus
printif(f"Empty GPUs: {empty_gpus}")
# get price that is being wasted
price = 0
for keys, values in empty_gpus.items():
    for key in PRICES:
        if key in keys:
            price += values * PRICES[key]
printif(f"{price:1f}$/hour is being wasted!")

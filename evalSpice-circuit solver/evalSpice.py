import numpy as np
from collections import defaultdict

# Function to evaluate a circuit in `msg`
def evalSpice(file_name):
      # Create a string file object
    
    elements = defaultdict(list)
    #elements = defaultdict(list)  # Stores elements (R, I, V) and their data
    resistors = dict()  # Stores resistance components
    voltages = dict()  # Stores voltage sources
    currents = dict()  # Stores current sources
    node_list = []  # Stores all unique nodes
    valid_elements = ['I', 'V', 'R'] # Stores all the valid element names
    circuit_start = False
    circuit_end = False
    voltage_names=[]
    try:
        with open(file_name, "r") as file: #Opening a file
            for line in file:
                # Skipping lines and also stripping the white spaces until the ".circuit" line is found
                if not circuit_start:
                    if line.strip() == '.circuit':
                        circuit_start = True
                    continue
                line = line.strip().split()

                if line and line[0][0] != '.':
                    element_type = line[0][0]  # First character indicates element type (R, I, V)
                    element_data = []  # Initialize an empty list to store the modified elements

                    # Iterate over each element (x) in line[1:]
                    for x in line[1:]:
                        if x == 'GND':
                            element_data.append('0')  # If x is 'GND', append '0' to element_data
                        else:
                            element_data.append(x)   # If x is not 'GND', append x to element_data

                    node_list.extend(element_data[:2])
                    
                    elements[element_type].append(element_data)  # Storing the element data
                    if element_type not in valid_elements:
                            raise ValueError('Only V, I, R elements are permitted')
                    if element_type=='V':
                        voltage_names.append(line[0])

                elif line and line[0] == '.end':
                    if not circuit_start or circuit_end:
                        raise ValueError("Malformed circuit file")
                    circuit_end = True
                    break

            if not circuit_start or not circuit_end:
                raise ValueError("Malformed circuit file")
                
            node_list = list(set(node_list))
            node_list.sort()  # Sorting the  nodes in ascending order
            
            # Processing the resistors in each branch
            for each_branch in elements['R']:
                node1, node2, resistance = each_branch[0], each_branch[1], each_branch[2]
                resistors[(node1, node2)] = float(resistance)
                
            # Processing the current sources in each branch 
            for each_branch in elements['I']:
                node1, node2, current = each_branch[0], each_branch[1], each_branch[3]
                currents[(node1, node2)] = float(current)
                
            # Processing voltage sources in each branch
            for each_branch in elements['V']:
                node1, node2, voltage = each_branch[0], each_branch[1], each_branch[3]
                voltages[(node1, node2)] = float(voltage)
            
            # Creating a dictionary to map node names to their indices
            dict_of_nodes = {node: i for i, node in enumerate(node_list)}
            
            
            # Creating a conductance and current matrix
            conductance, current_matrix = conductance_matrix(len(dict_of_nodes), resistors, voltages, currents, dict_of_nodes)
            
            # Solving for node voltages using linalg, even Gaussian Elimination could have been used
            try:
                V = np.linalg.solve(conductance, current_matrix)
            except np.linalg.LinAlgError:
                raise ValueError('Circuit error: no solution')
                    
            Node_V = {Node: V[dict_of_nodes[Node]-1] for Node in node_list if Node != '0'}  # Create a dictionary for node voltages
            Node_V['0'] = 0.0  # Set ground voltage to 0
            i_dict=dict()
            i_dict= {source_name: V[len(V) + i - 1]
                for i, source_name in enumerate(voltage_names)
            }
            Node_V['GND'] = Node_V.pop('0')
        return Node_V,i_dict
    except FileNotFoundError:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")
        

# Function to create the conductance and current matrices
def conductance_matrix(num_nodes, resistors, voltages, currents, nodenum):
    voltage_sources = len(voltages)
    conductance = np.zeros((num_nodes + voltage_sources, num_nodes + voltage_sources))
    current_matrix = np.zeros(num_nodes + voltage_sources)
    current_row = num_nodes
    
    # Processing resistors to create the matrix
    for (node1, node2), R in resistors.items():
        i, j = nodenum[node1], nodenum[node2]
        conductance[i, i] += 1 / R
        conductance[j, j] += 1 / R
        conductance[i, j] -= 1 / R
        conductance[j, i] -= 1 / R
    
    # Processing voltages to create the matrix
    for (node1, node2), voltage in voltages.items():
        i, j = nodenum[node1], nodenum[node2]
        conductance[current_row, i] += 1
        conductance[current_row, j] -= 1
        conductance[i, current_row] += 1
        conductance[j, current_row] -= 1
        current_matrix[current_row] += voltage
        current_row += 1
    
    # Processing currents to create the matrix
    for (node1, node2), current in currents.items():
        i, j = nodenum[node1], nodenum[node2]
        current_matrix[i] -= current
        current_matrix[j] += current
        
    return conductance[1:, 1:], current_matrix[1:]  # Returning the matrices

def create_matrices(num_nodes, resistances, voltage_sources, current_sources, node_indices):
    num_voltage_sources = len(voltage_sources)
    G = np.zeros((num_nodes + num_voltage_sources, num_nodes + num_voltage_sources))
    right_vector = np.zeros(num_nodes + num_voltage_sources)
    voltage_source_index = num_nodes

    for (node1, node2), R in resistances.items():
        i, j = node_indices[node1], node_indices[node2]
        G[i, i] += 1 / R
        G[j, j] += 1 / R
        G[i, j] -= 1 / R
        G[j, i] -= 1 / R
    
    for (node1, node2), voltage in voltage_sources.items():
        i, j = node_indices[node1], node_indices[node2]
        G[voltage_source_index, j] -= 1
        G[voltage_source_index, i] += 1
        G[i, voltage_source_index] += 1
        G[j, voltage_source_index] -= 1
        right_vector[voltage_source_index] += voltage
        voltage_source_index += 1

    for (node1, node2), current in current_sources.items():
        i, j = node_indices[node1], node_indices[node2]
        right_vector[i] -= current
        right_vector[j] += current
    return G[1:, 1:], right_vector[1:]
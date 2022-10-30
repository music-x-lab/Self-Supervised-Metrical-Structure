import numpy as np

def get_ternary_transition(del_panelty, ins_panelty):
    n_layers = len(del_panelty)
    z_to_tuple = []
    tuple_to_z = {}
    z_to_layer = []

    for z in range(3 ** n_layers):
        t = []
        for i in range(n_layers + 1):
            if (z % (3 ** (i + 1)) != 0 or i == n_layers):
                z_to_layer.append(i)
                break
        for i in range(n_layers):
            t.append(z % 3)
            z //= 3
        z_to_tuple.append(tuple(t))
        tuple_to_z[tuple(t)] = z
    trans = np.zeros((3 ** n_layers, 3 ** n_layers))
    for z in range(3 ** n_layers):
        t = z_to_tuple[z]
        for z2 in range(3 ** n_layers):
            t2 = z_to_tuple[z2]
            lower_level_cleared = True
            panelty = 0.0
            for i in range(n_layers):
                if (lower_level_cleared):
                    if (t[i] == 0 and t2[i] == 0):
                        panelty += del_panelty[i]
                    elif (t[i] == t2[i]):
                        panelty = -np.inf
                    elif (t[i] == 0 and t2[i] == 2):
                        panelty = -np.inf
                    elif (t[i] == 1 and t2[i] == 2):
                        panelty += ins_panelty[i]
                    elif (t[i] == 2 and t2[i] == 1):
                        panelty = -np.inf
                else:
                    if (t[i] != t2[i]):
                        panelty = -np.inf
                if (t2[i] != 0):
                    lower_level_cleared = False
            trans[z][z2] = panelty
    return trans, np.array(z_to_layer)

def get_binary_transition(del_panelty, ins_panelty):
    n_layers = len(del_panelty)
    z_to_tuple = []
    tuple_to_z = {}
    z_to_layer = []

    for z in range(2 ** n_layers):
        t = []
        for i in range(n_layers + 1):
            if (z % (2 ** (i + 1)) != 0 or i == n_layers):
                z_to_layer.append(i)
                break
        for i in range(n_layers):
            t.append(z % 2)
            z //= 2
        z_to_tuple.append(tuple(t))
        tuple_to_z[tuple(t)] = z
    trans = np.zeros((2 ** n_layers, 2 ** n_layers))
    for z in range(2 ** n_layers):
        t = z_to_tuple[z]
        for z2 in range(2 ** n_layers):
            t2 = z_to_tuple[z2]
            lower_level_cleared = True
            panelty = 0.0
            for i in range(n_layers):
                if (lower_level_cleared):
                    if (t[i] == 0 and t2[i] == 0):
                        panelty += del_panelty[i]
                    elif (t[i] == 1 and t2[i] == 1):
                        panelty += ins_panelty[i]
                else:
                    if (t[i] != t2[i]):
                        panelty = -np.inf
                if (t2[i] != 0):
                    lower_level_cleared = False
            trans[z][z2] = panelty
    return trans, np.array(z_to_layer)

if __name__ == '__main__':
    # perform some tests
    del_panelty = np.array([-30.0, -20.0, -10.0, -7.5, -5.0, -5.0, -5.0, -5.0])
    ins_panelty = np.array([-60.0, -40.0, -20.0, -10.0, -10.0, -10.0, -10.0, -10.0])
    trans_mat, z_to_layer = get_binary_transition(del_panelty, ins_panelty)
    print(trans_mat, z_to_layer)
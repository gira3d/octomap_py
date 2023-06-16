import octomap_py
from octomap_py import Point3D, ColorOcTree
from termcolor import cprint
from tqdm import tqdm

if __name__ == "__main__":
    res = 0.05
    tree = ColorOcTree(res)

    # insert some measurements of occupied cells
    occ_ext = 20
    for x in tqdm(range(-occ_ext, occ_ext, 1)):
        for y in range(-occ_ext, occ_ext, 1):
            for z in range(-occ_ext, occ_ext, 1):
                endpoint = Point3D(float(x) * 0.05 + 0.01,
                                   float(y) * 0.05 + 0.01,
                                   float(z) * 0.05 + 0.01)
                tree.update_node_and_color(endpoint, True,
                                           z * 5 + 100,
                                           x * 5 + 100,
                                           y * 5 + 100)

    # insert some measurements of free cells
    free_ext = 30
    for x in tqdm(range(-free_ext, free_ext, 1)):
        for y in range(-free_ext, free_ext, 1):
            for z in range(-free_ext, free_ext, 1):
                endpoint = Point3D(float(x) * 0.02 + 2.0 + 1e-6,
                                   float(y) * 0.02 + 2.0 + 1e-6,
                                   float(z) * 0.02 + 2.0 + 1e-6)
                tree.update_node_and_color(
                    endpoint, False, 255, 255, 0)

    tree.update_inner_occupancy()

    print(tree.size())
    initial_size = tree.size()
    assert (initial_size == 1034)
    assert (tree.size() == tree.calc_num_nodes())
    cprint('tree size test passed.', 'green')

    # manually pruning has no effect, should already be pruned
    tree.prune()
    assert (tree.size() == tree.calc_num_nodes())
    cprint('tree size test after pruning passed.', 'green')

    tree.write("color_tree.ot")
    read_tree = octomap_py.read("color_tree.ot")
    assert (read_tree.size() == tree.size())
    cprint('tree read/write test passed.', 'green')

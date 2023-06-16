import numpy as np
import open3d as o3d
from termcolor import cprint

from octomap_py import ColorOcTree
from sogmm_py.utils import np_to_o3d, o3d_to_np

if __name__ == "__main__":
    res = 0.005
    tree = ColorOcTree(res)

    input_pcld = o3d_to_np(o3d.io.read_point_cloud('../test_data/test.pcd', format='pcd'))
    cprint('input pcld size %d' % (input_pcld.shape[0]), 'green')

    tree.insert_color_occ_points(input_pcld)
    tree.update_inner_occupancy()

    tree.write('test.ot')

    ot_pcld = np_to_o3d(tree.get_color_occ_points())

    o3d.visualization.draw_geometries([ot_pcld])
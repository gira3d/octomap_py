#include <algorithm>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace om = octomap;
namespace omath = octomath;
namespace py = pybind11;

PYBIND11_MODULE(octomap_py, m)
{
  py::class_<om::point3d>(m, "Point3D")
      .def(py::init<>())
      .def(py::init<om::point3d>())
      .def(py::init<float, float, float>());

  py::class_<om::ColorOcTreeNode::Color>(m, "Color")
      .def(py::init<>())
      .def(py::init<uint8_t, uint8_t, uint8_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def_readwrite("r", &om::ColorOcTreeNode::Color::r)
      .def_readwrite("g", &om::ColorOcTreeNode::Color::g)
      .def_readwrite("b", &om::ColorOcTreeNode::Color::b);

  py::class_<om::ColorOcTreeNode>(m, "ColorOcTreeNode")
      .def(py::init<>())
      .def(py::init<om::ColorOcTreeNode>())
      .def("get_color", py::overload_cast<>(&om::ColorOcTreeNode::getColor))
      .def("set_color", py::overload_cast<om::ColorOcTreeNode::Color>(
                            &om::ColorOcTreeNode::setColor))
      .def("set_color", py::overload_cast<uint8_t, uint8_t, uint8_t>(
                            &om::ColorOcTreeNode::setColor));

  py::class_<om::ColorOcTree>(m, "ColorOcTree")
      .def(py::init<double>())
      .def("set_occupancy_thres", &om::ColorOcTree::setOccupancyThres)
      .def("set_prob_hit", &om::ColorOcTree::setProbHit)
      .def("set_prob_miss", &om::ColorOcTree::setProbMiss)
      .def("set_clamping_thres_min", &om::ColorOcTree::setClampingThresMin)
      .def("set_clamping_thres_max", &om::ColorOcTree::setClampingThresMax)
      .def("get_occupancy_thres", &om::ColorOcTree::getOccupancyThres)
      .def("get_prob_hit", &om::ColorOcTree::getProbHit)
      .def("get_prob_miss", &om::ColorOcTree::getProbMiss)
      .def("get_clamping_thres_min", &om::ColorOcTree::getClampingThresMin)
      .def("get_clamping_thres_max", &om::ColorOcTree::getClampingThresMax)
      .def("get_resolution", &om::ColorOcTree::getResolution)
      .def("insert_color_occ_points",
           [](om::ColorOcTree& tree, const Eigen::MatrixXf& pcld) {
             for (unsigned int i = 0; i < pcld.rows(); i++)
             {
               om::point3d ep(pcld(i, 0), pcld(i, 1), pcld(i, 2));
               om::ColorOcTreeNode* ret = tree.updateNode(ep, true);
               // Convert [0.0, 1.0] color to [0, 255]
               // https://stackoverflow.com/a/1914172
               uint8_t g = std::max(
                   0, std::min(255, (int)std::floor(pcld(i, 3) * 256.0f)));
               ret->setColor(g, g, g);
             }
           })
      .def("get_color_occ_points",
           [](om::ColorOcTree& tree) {
             unsigned int max_depth = tree.getTreeDepth();
             std::vector<om::ColorOcTreeNode*> collapsed_occ_nodes;
             do
             {
               collapsed_occ_nodes.clear();
               for (om::ColorOcTree::iterator it = tree.begin();
                    it != tree.end(); ++it)
               {
                 if (tree.isNodeOccupied(*it) && it.getDepth() < max_depth)
                 {
                   collapsed_occ_nodes.push_back(&(*it));
                 }
               }
               for (std::vector<om::ColorOcTreeNode*>::iterator it =
                        collapsed_occ_nodes.begin();
                    it != collapsed_occ_nodes.end(); ++it)
               {
                 tree.expandNode(*it);
               }
             } while (collapsed_occ_nodes.size() > 0);

             std::vector<om::point3d> points;
             std::vector<om::ColorOcTreeNode::Color> colors;
             unsigned int size = 0;
             for (om::ColorOcTree::iterator it = tree.begin(); it != tree.end();
                  ++it)
             {
              if (tree.isNodeOccupied(*it))
              {
                points.push_back(it.getCoordinate());
                colors.push_back(it->getColor());
                size += 1;
              }
             }

             Eigen::MatrixXf pcld = Eigen::MatrixXf::Zero(size, 4);
             for (unsigned int i = 0; i < size; i++)
             {
              pcld(i, 0) = points[i].x();
              pcld(i, 1) = points[i].y();
              pcld(i, 2) = points[i].z();
              uint8_t r = colors[i].r;
              float c = std::max(0.0f, std::min(1.0f, (float) r / 255));
              pcld(i, 3) = c;
             }

             return pcld;
           })
      .def("enable_change_detection", &om::ColorOcTree::enableChangeDetection)
      .def("reset_change_detection", &om::ColorOcTree::resetChangeDetection)
      .def("num_changes_detected", &om::ColorOcTree::numChangesDetected)
      .def("begin", &om::ColorOcTree::begin)
      .def("end", &om::ColorOcTree::end)
      .def("update_inner_occupancy", &om::ColorOcTree::updateInnerOccupancy)
      .def("prune", &om::ColorOcTree::prune)
      .def("get_color_at_point", [](om::ColorOcTree& tree, const om::point3d& qp) {
        om::ColorOcTreeNode* result = tree.search(qp);
        om::ColorOcTreeNode::Color color = result->getColor();
        uint8_t r = color.r;
        return std::max(0.0f, std::min(1.0f, (float) r / 255));
      })
      .def("get_color_at_points", [](om::ColorOcTree& tree, const Eigen::MatrixXf& pcld) {
          Eigen::MatrixXf colors = Eigen::MatrixXf::Zero(pcld.rows(), 1);
          for (unsigned int i = 0; i < pcld.rows(); i++)
          {
            om::point3d ep(pcld(i, 0), pcld(i, 1), pcld(i, 2));
            om::ColorOcTreeNode* result = tree.search(ep);
            om::ColorOcTreeNode::Color color = result->getColor();
            uint8_t r = color.r;
            colors(i, 0) = std::max(0.0f, std::min(1.0f, (float) r / 255));
          }

          return colors;
      })
      .def("calc_num_nodes", &om::ColorOcTree::calcNumNodes)
      .def("size", &om::ColorOcTree::size)
      .def("write", py::overload_cast<const std::string&>(
                        &om::ColorOcTree::write, py::const_))
      .def("update_node", [](om::ColorOcTree& tree, const om::point3d& ep,
                             bool occ) { tree.updateNode(ep, occ); })
      .def("update_node_and_color",
           [](om::ColorOcTree& tree, const om::point3d& ep, bool occ, uint8_t r,
              uint8_t g, uint8_t b) {
             om::ColorOcTreeNode* ret = tree.updateNode(ep, occ);
             ret->setColor(r, g, b);
           });

  m.def("read", [](const std::string& filename) {
    om::AbstractOcTree* aot = om::AbstractOcTree::read(filename);
    om::ColorOcTree* cot = dynamic_cast<om::ColorOcTree*>(aot);

    return *cot;
  });
}
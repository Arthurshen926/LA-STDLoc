#include <PoseLib/camera_pose.h>
#include <PoseLib/robust/bundle.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/solvers/p3p.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using poselib::CameraPose;
using poselib::Point2D;
using poselib::Point3D;

namespace {

struct Result {
    Eigen::Matrix4d w2c = Eigen::Matrix4d::Identity();
    std::vector<int> inliers;
    size_t iterations = 0;
    size_t diverse_samples = 0;
    size_t fallback_samples = 0;
    size_t local_refinements = 0;
    bool rescue_used = false;
};

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double value = values[middle];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
        value = 0.5 * (value + values[middle - 1]);
    }
    return value;
}

class DependencySampler {
  public:
    DependencySampler(
        size_t count,
        const std::vector<double> &scores,
        double guided_mixture,
        double rank_power,
        uint64_t seed
    ) : rng_(seed), uniform_(0, count - 1) {
        if (!scores.empty() && guided_mixture > 0.0) {
            std::vector<size_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
                return scores[left] > scores[right];
            });
            weights_.assign(count, (1.0 - guided_mixture) / static_cast<double>(count));
            double guided_sum = 0.0;
            for (size_t rank = 1; rank <= count; ++rank) {
                guided_sum += 1.0 / std::pow(static_cast<double>(rank), rank_power);
            }
            for (size_t rank = 1; rank <= count; ++rank) {
                weights_[order[rank - 1]] += guided_mixture *
                    (1.0 / std::pow(static_cast<double>(rank), rank_power)) / guided_sum;
            }
            guided_ = std::discrete_distribution<size_t>(weights_.begin(), weights_.end());
        }
    }

    std::array<size_t, 3> draw() {
        std::array<size_t, 3> sample{};
        for (size_t k = 0; k < 3; ++k) {
            do {
                sample[k] = weights_.empty() ? uniform_(rng_) : guided_(rng_);
            } while (
                (k >= 1 && sample[k] == sample[0]) ||
                (k >= 2 && sample[k] == sample[1])
            );
        }
        return sample;
    }

  private:
    std::mt19937_64 rng_;
    std::uniform_int_distribution<size_t> uniform_;
    std::vector<double> weights_;
    std::discrete_distribution<size_t> guided_;
};

bool diverse(
    const std::array<size_t, 3> &sample,
    const std::vector<int64_t> &dependency_groups,
    const std::vector<int64_t> &image_cells,
    const std::vector<int64_t> &surface_groups,
    const std::vector<Point3D> &points3d,
    double scene_scale
) {
    std::unordered_set<int64_t> dependencies;
    std::unordered_set<int64_t> cells;
    std::unordered_set<int64_t> surfaces;
    for (size_t index : sample) {
        dependencies.insert(dependency_groups[index]);
        cells.insert(image_cells[index]);
        surfaces.insert(surface_groups[index]);
    }
    const Point3D e01 = points3d[sample[1]] - points3d[sample[0]];
    const Point3D e02 = points3d[sample[2]] - points3d[sample[0]];
    const Point3D e12 = points3d[sample[2]] - points3d[sample[1]];
    const double extent = std::max({e01.norm(), e02.norm(), e12.norm()});
    const double area = 0.5 * e01.cross(e02).norm();
    return dependencies.size() == 3 && cells.size() >= 3 && surfaces.size() >= 2 &&
        extent >= 0.02 * scene_scale && area >= 1e-4 * scene_scale * scene_scale;
}

void refine(
    const std::vector<Point2D> &points2d,
    const std::vector<Point3D> &points3d,
    double threshold,
    CameraPose *pose
) {
    poselib::BundleOptions options;
    options.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    options.loss_scale = threshold;
    options.max_iterations = 25;
    poselib::bundle_adjust(points2d, points3d, pose, options);
}

Result solve(
    const std::vector<Point2D> &pixels,
    const std::vector<Point3D> &points3d,
    const Eigen::Matrix3d &K,
    const std::vector<int64_t> &dependency_groups,
    const std::vector<int64_t> &image_cells,
    const std::vector<int64_t> &surface_groups,
    const std::vector<double> &sampling_scores,
    double guided_mixture,
    double guided_rank_power,
    double reprojection_error,
    double confidence,
    size_t max_iterations,
    size_t min_iterations,
    size_t rescue_max_iterations,
    double rescue_inlier_ratio,
    uint64_t seed
) {
    const size_t count = pixels.size();
    if (count != points3d.size() || count != dependency_groups.size() ||
        count != image_cells.size() || count != surface_groups.size()) {
        throw std::invalid_argument("correspondences and dependency metadata must align");
    }
    if (!sampling_scores.empty() && sampling_scores.size() != count) {
        throw std::invalid_argument("sampling scores must align with correspondences");
    }
    if (count < 4) {
        return {};
    }
    if (guided_rank_power <= 0.0) {
        throw std::invalid_argument("guided_rank_power must be positive");
    }

    std::vector<Point2D> calibrated(count);
    const Eigen::Matrix3d inverse_k = K.inverse();
    for (size_t i = 0; i < count; ++i) {
        const Eigen::Vector3d bearing =
            inverse_k * Eigen::Vector3d(pixels[i].x(), pixels[i].y(), 1.0);
        calibrated[i] = bearing.hnormalized();
    }
    const double focal = 0.5 * (std::abs(K(0, 0)) + std::abs(K(1, 1)));
    const double threshold = reprojection_error / std::max(focal, 1e-12);
    const double threshold_sq = threshold * threshold;

    Point3D center = Point3D::Zero();
    for (const Point3D &point : points3d) {
        center += point;
    }
    center /= static_cast<double>(count);
    std::vector<double> center_distances;
    center_distances.reserve(count);
    for (const Point3D &point : points3d) {
        center_distances.push_back((point - center).norm());
    }
    const double scene_scale = std::max(median(center_distances), 1e-6);

    DependencySampler sampler(
        count, sampling_scores, std::clamp(guided_mixture, 0.0, 1.0),
        guided_rank_power, seed
    );
    Result result;
    CameraPose best_pose;
    size_t best_count = 0;
    double best_cost = std::numeric_limits<double>::max();
    size_t target_iterations = max_iterations;
    const size_t hard_max = std::max(max_iterations, rescue_max_iterations);
    const double log_miss = std::log(
        1.0 - std::clamp(confidence, 1e-12, 1.0 - 1e-12)
    );

    while (result.iterations < std::max(min_iterations, target_iterations) &&
           result.iterations < hard_max) {
        std::array<size_t, 3> sample{};
        bool accepted = false;
        for (size_t attempt = 0; attempt < 32; ++attempt) {
            sample = sampler.draw();
            if (diverse(
                    sample, dependency_groups, image_cells, surface_groups,
                    points3d, scene_scale)) {
                accepted = true;
                ++result.diverse_samples;
                break;
            }
        }
        if (!accepted) {
            sample = sampler.draw();
            ++result.fallback_samples;
        }

        std::vector<Point3D> bearings(3);
        std::vector<Point3D> world(3);
        for (size_t k = 0; k < 3; ++k) {
            bearings[k] = calibrated[sample[k]].homogeneous().normalized();
            world[k] = points3d[sample[k]];
        }
        std::vector<CameraPose> models;
        poselib::p3p(bearings, world, &models);
        for (CameraPose &pose : models) {
            size_t inlier_count = 0;
            double cost = poselib::compute_msac_score(
                pose, calibrated, points3d, threshold_sq, &inlier_count
            );
            if (inlier_count > best_count ||
                (inlier_count == best_count && cost < best_cost)) {
                if (inlier_count >= 6) {
                    refine(calibrated, points3d, threshold, &pose);
                    ++result.local_refinements;
                    cost = poselib::compute_msac_score(
                        pose, calibrated, points3d, threshold_sq, &inlier_count
                    );
                }
                if (inlier_count > best_count ||
                    (inlier_count == best_count && cost < best_cost)) {
                    best_pose = pose;
                    best_count = inlier_count;
                    best_cost = cost;
                    const double ratio = static_cast<double>(best_count) /
                        static_cast<double>(count);
                    if (ratio > 0.0) {
                        const double all_inlier = std::clamp(
                            ratio * ratio * ratio, 1e-12, 1.0 - 1e-12
                        );
                        const size_t required = static_cast<size_t>(
                            std::ceil(log_miss / std::log(1.0 - all_inlier))
                        );
                        target_iterations = std::min(
                            max_iterations, std::max(min_iterations, required)
                        );
                    }
                }
            }
        }
        ++result.iterations;
        const double ratio = static_cast<double>(best_count) /
            static_cast<double>(count);
        if (result.iterations >= max_iterations &&
            hard_max > max_iterations && ratio < rescue_inlier_ratio) {
            target_iterations = hard_max;
            result.rescue_used = true;
        }
    }

    if (best_count < 4) {
        return result;
    }
    refine(calibrated, points3d, threshold, &best_pose);
    std::vector<char> mask;
    poselib::get_inliers(best_pose, calibrated, points3d, threshold_sq, &mask);
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            result.inliers.push_back(static_cast<int>(i));
        }
    }
    result.w2c.block<3, 4>(0, 0) = best_pose.Rt();
    return result;
}

}  // namespace

PYBIND11_MODULE(_lafgs_poselib, module) {
    module.doc() = "Compiled dependency-aware absolute-pose sampler for LaFGS";
    module.def(
        "solve_dependency_absolute_pose",
        [](const std::vector<Point2D> &points2d,
           const std::vector<Point3D> &points3d,
           const Eigen::Matrix3d &K,
           const std::vector<int64_t> &dependency_groups,
           const std::vector<int64_t> &image_cells,
           const std::vector<int64_t> &surface_groups,
           const std::vector<double> &sampling_scores,
           double guided_mixture,
           double guided_rank_power,
           double reprojection_error,
           double confidence,
           size_t max_iterations,
           size_t min_iterations,
           size_t rescue_max_iterations,
           double rescue_inlier_ratio,
           uint64_t seed) {
            Result result;
            {
                py::gil_scoped_release release;
                result = solve(
                    points2d, points3d, K, dependency_groups, image_cells,
                    surface_groups, sampling_scores, guided_mixture,
                    guided_rank_power, reprojection_error, confidence,
                    max_iterations, min_iterations, rescue_max_iterations,
                    rescue_inlier_ratio, seed
                );
            }
            py::dict diagnostics;
            diagnostics["iterations"] = result.iterations;
            diagnostics["diverse_samples"] = result.diverse_samples;
            diagnostics["fallback_samples"] = result.fallback_samples;
            diagnostics["local_refinements"] = result.local_refinements;
            diagnostics["rescue_used"] = result.rescue_used;
            diagnostics["backend"] = "cpp";
            return py::make_tuple(result.w2c, result.inliers, diagnostics);
        },
        py::arg("points2d"), py::arg("points3d"), py::arg("K"),
        py::arg("dependency_groups"), py::arg("image_cells"),
        py::arg("surface_groups"), py::arg("sampling_scores"),
        py::arg("guided_mixture"), py::arg("guided_rank_power"),
        py::arg("reprojection_error"), py::arg("confidence"),
        py::arg("max_iterations"), py::arg("min_iterations"),
        py::arg("rescue_max_iterations"), py::arg("rescue_inlier_ratio"),
        py::arg("seed")
    );
}

#include "lafgs_preemptive_pose.h"

#include <PoseLib/camera_pose.h>
#include <PoseLib/misc/colmap_models.h>
#include <PoseLib/robust/bundle.h>
#include <PoseLib/robust/sampling.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/solvers/p3p.h>
#include <PoseLib/types.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using poselib::Camera;
using poselib::CameraPose;
using poselib::Point2D;
using poselib::Point3D;
using poselib::RansacOptions;
using poselib::RansacStats;

namespace {

constexpr const char *kPreemptiveSolverVersion =
    "preemptive_poselib_exact_v1";

struct ScoreResult {
    double score = std::numeric_limits<double>::max();
    size_t inlier_count = 0;
    bool pruned = false;
};

struct VerificationCounters {
    size_t score_calls = 0;
    size_t complete_score_calls = 0;
    size_t pruned_score_calls = 0;
    size_t residual_evaluations = 0;
};

class PreemptiveAbsolutePoseEstimator {
  public:
    PreemptiveAbsolutePoseEstimator(
        const RansacOptions &ransac_options,
        const std::vector<Point2D> &points2d,
        const std::vector<Point3D> &points3d,
        const std::vector<double> &verification_priorities,
        size_t check_interval
    )
        : num_data(points2d.size()), opt_(ransac_options), x_(points2d),
          X_(points3d),
          sampler_(num_data, sample_sz, opt_.seed, opt_.progressive_sampling,
                   opt_.max_prosac_iterations),
          check_interval_(std::max<size_t>(check_interval, 1)),
          ordered_x_(num_data), ordered_X_(num_data),
          inlier_residuals_(num_data), inlier_mask_(num_data) {
        xs_.resize(sample_sz);
        Xs_.resize(sample_sz);
        sample_.resize(sample_sz);
        verification_order_.resize(num_data);
        for (size_t index = 0; index < num_data; ++index) {
            verification_order_[index] = index;
        }
        if (!verification_priorities.empty()) {
            if (verification_priorities.size() != num_data) {
                throw std::invalid_argument(
                    "verification priorities must align with correspondences"
                );
            }
            std::stable_sort(
                verification_order_.begin(), verification_order_.end(),
                [&](size_t left, size_t right) {
                    const double left_value =
                        std::isfinite(verification_priorities[left])
                            ? verification_priorities[left]
                            : -std::numeric_limits<double>::infinity();
                    const double right_value =
                        std::isfinite(verification_priorities[right])
                            ? verification_priorities[right]
                            : -std::numeric_limits<double>::infinity();
                    return left_value > right_value;
                }
            );
        }
        for (size_t position = 0; position < num_data; ++position) {
            const size_t original_index = verification_order_[position];
            ordered_x_[position] = x_[original_index];
            ordered_X_[position] = X_[original_index];
        }
    }

    void generate_models(std::vector<CameraPose> *models) {
        sampler_.generate_sample(&sample_);
        for (size_t index = 0; index < sample_sz; ++index) {
            xs_[index] = x_[sample_[index]].homogeneous().normalized();
            Xs_[index] = X_[sample_[index]];
        }
        poselib::p3p(xs_, Xs_, models);
    }

    void refine_model(CameraPose *pose) const {
        poselib::BundleOptions options;
        options.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
        options.loss_scale = opt_.max_reproj_error;
        options.max_iterations = 25;
        poselib::bundle_adjust(x_, X_, pose, options);
    }

    ScoreResult score_model(
        const CameraPose &pose,
        double score_limit,
        size_t inlier_limit,
        bool guard_inlier_trigger
    ) const {
        ++counters_.score_calls;
        const double threshold_sq =
            opt_.max_reproj_error * opt_.max_reproj_error;
        const Eigen::Matrix3d rotation = pose.R();
        const double p00 = rotation(0, 0);
        const double p01 = rotation(0, 1);
        const double p02 = rotation(0, 2);
        const double p03 = pose.t(0);
        const double p10 = rotation(1, 0);
        const double p11 = rotation(1, 1);
        const double p12 = rotation(1, 2);
        const double p13 = pose.t(1);
        const double p20 = rotation(2, 0);
        const double p21 = rotation(2, 1);
        const double p22 = rotation(2, 2);
        const double p23 = pose.t(2);

        double partial_score = 0.0;
        size_t inlier_count = 0;
        size_t evaluated = 0;
        for (size_t begin = 0; begin < num_data;
             begin += check_interval_) {
            const size_t end = std::min(begin + check_interval_, num_data);
            for (size_t position = begin; position < end; ++position) {
                const size_t ordered_index = verification_order_[position];
                const Point3D &point3d = ordered_X_[position];
                const Point2D &point2d = ordered_x_[position];
                const double z0 = p00 * point3d(0) + p01 * point3d(1) +
                                  p02 * point3d(2) + p03;
                const double z1 = p10 * point3d(0) + p11 * point3d(1) +
                                  p12 * point3d(2) + p13;
                const double z2 = p20 * point3d(0) + p21 * point3d(1) +
                                  p22 * point3d(2) + p23;
                const double inverse_z = 1.0 / z2;
                const double residual_x = z0 * inverse_z - point2d(0);
                const double residual_y = z1 * inverse_z - point2d(1);
                const double residual_sq =
                    residual_x * residual_x + residual_y * residual_y;
                if (residual_sq < threshold_sq && z2 > 0.0) {
                    inlier_mask_[ordered_index] = 1;
                    inlier_residuals_[ordered_index] = residual_sq;
                    partial_score += residual_sq;
                    ++inlier_count;
                } else {
                    inlier_mask_[ordered_index] = 0;
                    partial_score += threshold_sq;
                }
            }
            evaluated = end;
            counters_.residual_evaluations += end - begin;
            if (!std::isfinite(score_limit)) {
                continue;
            }
            const size_t maximum_inlier_count =
                inlier_count + (num_data - evaluated);
            const double tolerance =
                std::numeric_limits<double>::epsilon() *
                std::max(std::abs(score_limit), threshold_sq) *
                static_cast<double>(64 * std::max<size_t>(num_data, 1));
            const bool cannot_improve_score =
                partial_score > score_limit + tolerance;
            const bool cannot_trigger_inlier_refinement =
                !guard_inlier_trigger || maximum_inlier_count <= inlier_limit;
            if (cannot_improve_score && cannot_trigger_inlier_refinement) {
                ++counters_.pruned_score_calls;
                return {
                    partial_score,
                    inlier_count,
                    true,
                };
            }
        }

        // PoseLib accumulates only inlier residuals in original row order and
        // adds the outlier truncation cost once. Preserve that exact order.
        double score = 0.0;
        for (size_t index = 0; index < num_data; ++index) {
            if (inlier_mask_[index]) {
                score += inlier_residuals_[index];
            }
        }
        score += static_cast<double>(num_data - inlier_count) * threshold_sq;
        ++counters_.complete_score_calls;
        return {score, inlier_count, false};
    }

    const VerificationCounters &counters() const { return counters_; }

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt_;
    const std::vector<Point2D> &x_;
    const std::vector<Point3D> &X_;
    poselib::RandomSampler sampler_;
    size_t check_interval_;
    std::vector<Point3D> xs_;
    std::vector<Point3D> Xs_;
    std::vector<size_t> sample_;
    std::vector<size_t> verification_order_;
    std::vector<Point2D> ordered_x_;
    std::vector<Point3D> ordered_X_;
    mutable std::vector<double> inlier_residuals_;
    mutable std::vector<char> inlier_mask_;
    mutable VerificationCounters counters_;
};

struct RansacState {
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score =
        std::numeric_limits<double>::max();
    size_t dynamic_max_iterations = 100000;
    double log_probability_missing_model = std::log(1.0 - 0.9999);
};

void score_models(
    PreemptiveAbsolutePoseEstimator *estimator,
    const std::vector<CameraPose> &models,
    const RansacOptions &options,
    RansacState *state,
    RansacStats *stats,
    CameraPose *best_model
) {
    int best_model_index = -1;
    for (size_t index = 0; index < models.size(); ++index) {
        const ScoreResult scored = estimator->score_model(
            models[index], state->best_minimal_msac_score,
            state->best_minimal_inlier_count, true
        );
        if (scored.pruned) {
            continue;
        }
        const bool more_inliers =
            scored.inlier_count > state->best_minimal_inlier_count;
        const bool better_score =
            scored.score < state->best_minimal_msac_score;
        if (more_inliers || better_score) {
            if (more_inliers) {
                state->best_minimal_inlier_count = scored.inlier_count;
            }
            if (better_score) {
                state->best_minimal_msac_score = scored.score;
            }
            best_model_index = static_cast<int>(index);
            if (scored.score < stats->model_score) {
                stats->model_score = scored.score;
                *best_model = models[index];
                stats->num_inliers = scored.inlier_count;
            }
        }
    }
    if (best_model_index < 0) {
        return;
    }

    CameraPose refined_model = models[best_model_index];
    estimator->refine_model(&refined_model);
    ++stats->refinements;
    const ScoreResult refined = estimator->score_model(
        refined_model, stats->model_score, 0, false
    );
    if (!refined.pruned && refined.score < stats->model_score) {
        stats->model_score = refined.score;
        stats->num_inliers = refined.inlier_count;
        *best_model = refined_model;
    }

    stats->inlier_ratio = static_cast<double>(stats->num_inliers) /
                          static_cast<double>(estimator->num_data);
    if (stats->inlier_ratio >= 0.9999) {
        state->dynamic_max_iterations = options.min_iterations;
    } else if (stats->inlier_ratio <= 0.0001) {
        state->dynamic_max_iterations = options.max_iterations;
    } else {
        const double outlier_probability =
            1.0 - std::pow(stats->inlier_ratio, estimator->sample_sz);
        state->dynamic_max_iterations = static_cast<size_t>(std::ceil(
            state->log_probability_missing_model /
            std::log(outlier_probability) * options.dyn_num_trials_mult
        ));
    }
}

RansacStats run_preemptive_ransac(
    PreemptiveAbsolutePoseEstimator *estimator,
    const RansacOptions &options,
    CameraPose *best_model
) {
    RansacStats stats;
    if (estimator->num_data < estimator->sample_sz) {
        return stats;
    }
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    RansacState state;
    state.dynamic_max_iterations = options.max_iterations;
    state.log_probability_missing_model =
        std::log(1.0 - options.success_prob);

    std::vector<CameraPose> models;
    for (stats.iterations = 0;
         stats.iterations < options.max_iterations;
         ++stats.iterations) {
        if (stats.iterations > options.min_iterations &&
            stats.iterations > state.dynamic_max_iterations) {
            break;
        }
        models.clear();
        estimator->generate_models(&models);
        score_models(
            estimator, models, options, &state, &stats, best_model
        );
    }

    CameraPose refined_model = *best_model;
    estimator->refine_model(&refined_model);
    ++stats.refinements;
    const ScoreResult refined = estimator->score_model(
        refined_model, stats.model_score, 0, false
    );
    if (!refined.pruned && refined.score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = refined.inlier_count;
    }
    return stats;
}

struct PreemptiveResult {
    Eigen::Matrix4d pose_w2c = Eigen::Matrix4d::Identity();
    std::vector<int> inliers;
    RansacStats stats;
    VerificationCounters counters;
};

PreemptiveResult solve_preemptive_absolute_pose(
    const std::vector<Point2D> &points2d,
    const std::vector<Point3D> &points3d,
    const Eigen::Matrix3d &intrinsics,
    const std::vector<double> &verification_priorities,
    double reprojection_error,
    double confidence,
    size_t max_iterations,
    size_t min_iterations,
    bool progressive_sampling,
    size_t max_prosac_iterations,
    size_t check_interval,
    unsigned long seed
) {
    if (points2d.size() != points3d.size()) {
        throw std::invalid_argument("2D and 3D points must align");
    }
    if (points2d.size() < 4) {
        return {};
    }
    if (!verification_priorities.empty() &&
        verification_priorities.size() != points2d.size()) {
        throw std::invalid_argument(
            "verification priorities must align with correspondences"
        );
    }
    const double fx = intrinsics(0, 0);
    const double fy = intrinsics(1, 1);
    const double cx = intrinsics(0, 2);
    const double cy = intrinsics(1, 2);
    if (!intrinsics.allFinite() || fx <= 0.0 || fy <= 0.0) {
        throw std::invalid_argument("camera intrinsics must be finite");
    }
    if (!std::isfinite(reprojection_error) || reprojection_error <= 0.0) {
        throw std::invalid_argument(
            "reprojection error must be finite and positive"
        );
    }
    if (max_iterations == 0 || min_iterations > max_iterations) {
        throw std::invalid_argument(
            "iteration bounds must satisfy 0 < min <= max"
        );
    }

    const Camera camera(
        "PINHOLE", {fx, fy, cx, cy},
        static_cast<int>(std::round(2.0 * cx)),
        static_cast<int>(std::round(2.0 * cy))
    );
    std::vector<Point2D> calibrated(points2d.size());
    for (size_t index = 0; index < points2d.size(); ++index) {
        camera.unproject(points2d[index], &calibrated[index]);
    }
    RansacOptions options;
    options.max_iterations = max_iterations;
    options.min_iterations = min_iterations;
    options.success_prob = confidence;
    options.max_reproj_error = reprojection_error / camera.focal();
    options.progressive_sampling = progressive_sampling;
    options.max_prosac_iterations = max_prosac_iterations;
    options.seed = seed;

    CameraPose pose;
    pose.q << 1.0, 0.0, 0.0, 0.0;
    pose.t.setZero();
    PreemptiveAbsolutePoseEstimator estimator(
        options, calibrated, points3d, verification_priorities,
        check_interval
    );
    PreemptiveResult result;
    result.stats = run_preemptive_ransac(&estimator, options, &pose);
    result.counters = estimator.counters();

    std::vector<char> inlier_mask;
    poselib::get_inliers(
        pose, calibrated, points3d,
        options.max_reproj_error * options.max_reproj_error, &inlier_mask
    );
    for (size_t index = 0; index < inlier_mask.size(); ++index) {
        if (inlier_mask[index]) {
            result.inliers.push_back(static_cast<int>(index));
        }
    }

    if (result.stats.num_inliers > 3) {
        std::vector<Point2D> inlier_points2d;
        std::vector<Point3D> inlier_points3d;
        inlier_points2d.reserve(result.inliers.size());
        inlier_points3d.reserve(result.inliers.size());
        const double scale = 1.0 / camera.focal();
        Camera normalized_camera = camera;
        normalized_camera.rescale(scale);
        for (int index : result.inliers) {
            inlier_points2d.push_back(points2d[index] * scale);
            inlier_points3d.push_back(points3d[index]);
        }
        poselib::BundleOptions bundle_options;
        bundle_options.loss_scale = 0.5 * reprojection_error * scale;
        poselib::bundle_adjust(
            inlier_points2d, inlier_points3d, normalized_camera, &pose,
            bundle_options
        );
    }
    result.pose_w2c.block<3, 4>(0, 0) = pose.Rt();
    return result;
}

}  // namespace

void bind_preemptive_pose(py::module_ &module) {
    module.attr("PREEMPTIVE_SOLVER_VERSION") = kPreemptiveSolverVersion;
    module.def(
        "solve_preemptive_absolute_pose",
        [](const std::vector<Point2D> &points2d,
           const std::vector<Point3D> &points3d,
           const Eigen::Matrix3d &intrinsics,
           const std::vector<double> &verification_priorities,
           double reprojection_error,
           double confidence,
           size_t max_iterations,
           size_t min_iterations,
           bool progressive_sampling,
           size_t max_prosac_iterations,
           size_t check_interval,
           unsigned long seed) {
            PreemptiveResult result;
            {
                py::gil_scoped_release release;
                result = solve_preemptive_absolute_pose(
                    points2d, points3d, intrinsics,
                    verification_priorities, reprojection_error,
                    confidence, max_iterations, min_iterations,
                    progressive_sampling, max_prosac_iterations,
                    check_interval, seed
                );
            }
            const size_t full_evaluations =
                result.counters.score_calls * points2d.size();
            py::dict diagnostics;
            diagnostics["iterations"] = result.stats.iterations;
            diagnostics["refinements"] = result.stats.refinements;
            diagnostics["num_inliers"] = result.stats.num_inliers;
            diagnostics["inlier_ratio"] = result.stats.inlier_ratio;
            diagnostics["model_score"] = result.stats.model_score;
            diagnostics["score_calls"] = result.counters.score_calls;
            diagnostics["complete_score_calls"] =
                result.counters.complete_score_calls;
            diagnostics["pruned_score_calls"] =
                result.counters.pruned_score_calls;
            diagnostics["residual_evaluations"] =
                result.counters.residual_evaluations;
            diagnostics["full_residual_evaluations"] = full_evaluations;
            diagnostics["residual_evaluation_reduction"] =
                full_evaluations > 0
                    ? 1.0 - static_cast<double>(
                                result.counters.residual_evaluations
                            ) /
                                static_cast<double>(full_evaluations)
                    : 0.0;
            diagnostics["implementation_version"] =
                kPreemptiveSolverVersion;
            diagnostics["backend"] = "cpp";
            return py::make_tuple(
                result.pose_w2c, result.inliers, diagnostics
            );
        },
        py::arg("points2d"), py::arg("points3d"),
        py::arg("intrinsics"), py::arg("verification_priorities"),
        py::arg("reprojection_error"), py::arg("confidence"),
        py::arg("max_iterations"), py::arg("min_iterations"),
        py::arg("progressive_sampling"),
        py::arg("max_prosac_iterations"), py::arg("check_interval"),
        py::arg("seed")
    );
}

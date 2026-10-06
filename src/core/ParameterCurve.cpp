#include "ParameterCurve.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>

namespace sv
{
namespace
{
/**
 * @brief Computes unsigned distance between two Blick positions as a double.
 *
 * Casts to uint64_t before subtracting to ensure correct wraparound math even
 * if positions span across negative or large signed Blick offsets.
 */
double blickDistance(Blick later, Blick earlier)
{
    // Ordered signed positions may span more than the signed Blick range.
    return static_cast<double>(static_cast<std::uint64_t>(later) - static_cast<std::uint64_t>(earlier));
}

/**
 * @brief Evaluates custom cubic spline interpolation between points[rightIndex - 1] and points[rightIndex].
 *
 * In Synthesizer V, cubic interpolation uses Hermite cubic splines with custom tangent calculations:
 *
 * 1. Surrounding values:
 *    - left: point at rightIndex - 1
 *    - right: point at rightIndex
 *    - precedingValue: point at rightIndex - 2 (or left.value if at left boundary)
 *    - followingValue: point at rightIndex + 1 (or right.value if at right boundary)
 *
 * 2. Spacing weighting:
 *    - SV calculates spacing relative to a quarter note:
 *      spacing = max(0.0, interval / blicksPerQuarter - 0.25)
 *    - Weight:
 *      weight = (spacing <= 4.0) ? 1.0 / (spacing * 10.0 + 1.0) : 0.0
 *
 *    Rationale:
 *    - For closely spaced points (interval <= 0.25 beats), weight = 1.0: tangents are determined
 *      by central differences across neighbors (Catmull-Rom style), producing smooth curves.
 *    - For widely spaced points (interval > 4.25 beats), weight = 0.0: tangents default purely
 *      to the chord slope (right - left)/interval, causing the curve to act like a smoothstep S-curve
 *      and preventing extreme overshoot/ringing over long musical bars.
 *
 * 3. Cubic Hermite polynomial basis functions:
 *    Let t = position (in range [0, 1]):
 *    h00(t) = 2*t^3 - 3*t^2 + 1
 *    h10(t) = t^3 - 2*t^2 + t
 *    h01(t) = -2*t^3 + 3*t^2
 *    h11(t) = t^3 - t^2
 *
 *    interpolated = h00(t)*p0 + h10(t)*m0 + h01(t)*p1 + h11(t)*m1
 *    where m0 = leftTangent, m1 = rightTangent (scaled by interval).
 */
double sampleCubic(const ParameterCurve& curve, std::size_t rightIndex, double interval, double position)
{
    const auto& points = curve.points;
    const auto& left = points[rightIndex - 1];
    const auto& right = points[rightIndex];

    const double precedingValue = rightIndex > 1 ? points[rightIndex - 2].value : left.value;
    const double followingValue = rightIndex + 1 < points.size() ? points[rightIndex + 1].value : right.value;

    const double precedingInterval = rightIndex > 1 ? blickDistance(right.position, points[rightIndex - 2].position) : interval + 1.0;
    const double followingInterval = rightIndex + 1 < points.size() ? blickDistance(points[rightIndex + 1].position, left.position) : interval + 1.0;

    // Adaptive tangent blending weight based on quarter-note beat distance
    const double spacing = std::max(0.0, interval / static_cast<double>(blicksPerQuarter) - 0.25);
    const double weight = spacing <= 4.0 ? 1.0 / (spacing * 10.0 + 1.0) : 0.0;

    // Base segment slope (chord secant)
    const double segmentSlope = (right.value - left.value) / interval * (1.0 - weight);

    // Blended left and right tangents
    const double leftTangent = (right.value - precedingValue) / precedingInterval * interval * weight + segmentSlope;
    const double rightTangent = (followingValue - left.value) / followingInterval * interval * weight + segmentSlope;

    // Hermite basis evaluation
    const double squared = position * position;
    const double cubed = squared * position;
    return (2.0 * cubed - 3.0 * squared + 1.0) * left.value
         + (position + cubed - 2.0 * squared) * leftTangent
         + (3.0 * squared - 2.0 * cubed) * right.value
         + (cubed - squared) * rightTangent;
}
} // namespace

double sampleParameterCurve(const ParameterCurve& curve, Blick position, double defaultValue)
{
    // Return default if no points are defined
    if (curve.points.empty())
    {
        return defaultValue;
    }

    const auto& points = curve.points;

    // Constant hold before the first point
    if (position <= points.front().position)
    {
        return points.front().value;
    }

    // Constant hold after the last point
    if (position >= points.back().position)
    {
        return points.back().value;
    }

    // Binary search for the first point with position >= query position
    const auto right = std::lower_bound(points.begin(), points.end(), position, [](const AutomationPoint& point, Blick value)
                                        { return point.position < value; });

    // Exact hit
    if (right->position == position)
    {
        return right->value;
    }

    // Interval between adjacent enclosing points [left, right]
    const auto& left = *(right - 1);
    const double interval = blickDistance(right->position, left.position);
    double fraction = blickDistance(position, left.position) / interval; // Normalized fraction in (0, 1)

    // Cubic Hermite interpolation
    if (curve.mode == "cubic" && points.size() > 2)
    {
        return sampleCubic(curve, static_cast<std::size_t>(right - points.begin()), interval, fraction);
    }

    // Cosine easing: fraction mapped to (1 - cos(pi * t)) / 2
    if (curve.mode == "cosine")
    {
        fraction = (1.0 - std::cos(fraction * std::numbers::pi)) * 0.5;
    }
    else if (curve.mode != "linear" && curve.mode != "cubic")
    {
        throw std::invalid_argument("Unsupported parameter curve interpolation mode: " + curve.mode);
    }

    // Linear blending using normalized (or cosine-warped) fraction
    return left.value * (1.0 - fraction) + right->value * fraction;
}

} // namespace sv

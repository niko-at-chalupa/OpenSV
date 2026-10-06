#pragma once

#include "Project.h"

namespace sv
{
/**
 * @brief Evaluates an automation ParameterCurve at a specific musical position (in blicks).
 *
 * Curve semantics and invariants:
 * - Control points must be sorted ascending by position with unique positions (enforced by normaliseProject).
 * - Empty curves return @p defaultValue.
 * - Positions before the first control point return the first point's value (sample-and-hold clamping).
 * - Positions after the last control point return the last point's value (sample-and-hold clamping).
 * - For positions between points, interpolation depends on `curve.mode`:
 *   - "linear": Piecewise linear interpolation.
 *   - "cosine": Half-cosine smoothstep easing: fraction = (1 - cos(pi * fraction)) / 2.
 *   - "cubic": Synthesizer V's custom Hermite cubic spline with adaptive spacing-weighted
 *              tangents that smoothly transition between Catmull-Rom and smoothstep.
 *
 * @param curve The automation curve to evaluate.
 * @param position Musical time position in blicks.
 * @param defaultValue Fallback value if the curve contains no points.
 * @return The interpolated floating-point value.
 */
[[nodiscard]] double sampleParameterCurve(const ParameterCurve& curve, Blick position, double defaultValue = 0.0);
} // namespace sv

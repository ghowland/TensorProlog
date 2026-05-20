// ============================================================
// src/builtins/stats.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const dispatch_mod = @import("dispatch.zig");
const linalg = @import("linalg.zig");

const Q16 = q16_mod.Q16;
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;
const intSqrt = linalg.intSqrt;

pub fn statsMean(data: []const Q16, n: i32) Q16 {
    const count: usize = @intCast(n);
    if (count == 0) return Q16.zero();
    var sum: i64 = 0;
    for (data[0..count]) |v| {
        sum += @intCast(v.v);
    }
    const ni: i64 = @intCast(n);
    return .{
        .v = @intCast(@divTrunc(sum, ni)),
        .r0 = @intCast(@mod(sum, ni)),
    };
}

pub fn statsVariance(data: []const Q16, n: i32) Q16 {
    const count: usize = @intCast(n);
    if (count <= 1) return Q16.zero();
    const mean = statsMean(data, n);
    const mean_v: i64 = @intCast(mean.v);
    var sum_sq: i64 = 0;
    for (data[0..count]) |v| {
        const diff: i64 = @as(i64, @intCast(v.v)) - mean_v;
        sum_sq += @divTrunc(diff * diff, @as(i64, Q16.D));
    }
    const ni: i64 = @intCast(n);
    return .{
        .v = @intCast(@divTrunc(sum_sq, ni)),
        .r0 = @intCast(@mod(sum_sq, ni)),
    };
}

pub fn statsMedian(data: []Q16, n: i32) Q16 {
    const count: usize = @intCast(n);
    if (count == 0) return Q16.zero();

    for (0..count) |i| {
        var min_idx = i;
        for (i + 1..count) |j| {
            if (Q16.compare(data[j], data[min_idx]) < 0) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            const tmp = data[i];
            data[i] = data[min_idx];
            data[min_idx] = tmp;
        }
    }

    if (count % 2 == 1) {
        return data[count / 2];
    }

    const a: i64 = @intCast(data[count / 2 - 1].v);
    const b: i64 = @intCast(data[count / 2].v);
    return .{
        .v = @intCast(@divTrunc(a + b, 2)),
        .r0 = @intCast(@mod(a + b, 2)),
    };
}

pub fn statsBayes(prior: []const Q16, likelihood: []const Q16, posterior: []Q16, n_hypotheses: i32) void {
    const nh: usize = @intCast(n_hypotheses);
    if (nh == 0) return;

    var evidence: i64 = 0;
    for (0..nh) |i| {
        const p: i64 = @intCast(prior[i].v);
        const l: i64 = @intCast(likelihood[i].v);
        evidence += @divTrunc(p * l, @as(i64, Q16.D));
    }

    if (evidence == 0) {
        for (0..nh) |i| {
            posterior[i] = Q16.zero();
        }
        return;
    }

    var sum: i64 = 0;
    for (0..nh) |i| {
        const p: i64 = @intCast(prior[i].v);
        const l: i64 = @intCast(likelihood[i].v);
        const numerator: i64 = @divTrunc(p * l, @as(i64, Q16.D));
        const post_v: i64 = @divTrunc(numerator * @as(i64, Q16.D), evidence);
        posterior[i] = .{ .v = @intCast(post_v), .r0 = 0 };
        sum += post_v;
    }

    const diff: i64 = @as(i64, Q16.D) - sum;
    if (diff != 0 and nh > 0) {
        var max_idx: usize = 0;
        var max_val: i32 = posterior[0].v;
        for (1..nh) |i| {
            if (posterior[i].v > max_val) {
                max_val = posterior[i].v;
                max_idx = i;
            }
        }
        posterior[max_idx].v += @intCast(diff);
    }
}

pub fn statsNormalize(data: []Q16, n: i32) void {
    const count: usize = @intCast(n);
    if (count == 0) return;

    var sum: i64 = 0;
    for (data[0..count]) |v| {
        sum += @intCast(v.v);
    }

    if (sum == 0) {
        if (count > 0) {
            data[0] = .{ .v = Q16.D, .r0 = 0 };
            for (1..count) |i| {
                data[i] = Q16.zero();
            }
        }
        return;
    }

    var new_sum: i64 = 0;
    for (0..count) |i| {
        const vi: i64 = @intCast(data[i].v);
        const scaled: i64 = @divTrunc(vi * @as(i64, Q16.D), sum);
        data[i] = .{ .v = @intCast(scaled), .r0 = 0 };
        new_sum += scaled;
    }

    const remainder: i64 = @as(i64, Q16.D) - new_sum;
    if (remainder != 0 and count > 0) {
        var max_idx: usize = 0;
        var max_v: i32 = data[0].v;
        for (1..count) |i| {
            if (data[i].v > max_v) {
                max_v = data[i].v;
                max_idx = i;
            }
        }
        data[max_idx].v += @intCast(remainder);
    }
}

pub fn statsHistogram(data: []const Q16, bin_edges: []const Q16, counts: []i32, n_data: i32, n_bins: i32) void {
    const nd: usize = @intCast(n_data);
    const nb: usize = @intCast(n_bins);

    for (0..nb) |i| {
        counts[i] = 0;
    }

    for (data[0..nd]) |val| {
        var placed = false;
        for (0..nb) |b| {
            const edge_idx = b + 1;
            if (edge_idx < bin_edges.len and Q16.compare(val, bin_edges[edge_idx]) < 0) {
                counts[b] += 1;
                placed = true;
                break;
            }
        }
        if (!placed and nb > 0) {
            counts[nb - 1] += 1;
        }
    }
}

pub fn statsCorrelation(x: []const Q16, y: []const Q16, n: i32) Q16 {
    const count: usize = @intCast(n);
    if (count <= 1) return Q16.zero();

    const mean_x = statsMean(x, n);
    const mean_y = statsMean(y, n);
    const mx: i64 = @intCast(mean_x.v);
    const my: i64 = @intCast(mean_y.v);

    var sum_xy: i64 = 0;
    var sum_xx: i64 = 0;
    var sum_yy: i64 = 0;

    for (0..count) |i| {
        const dx: i64 = @as(i64, @intCast(x[i].v)) - mx;
        const dy: i64 = @as(i64, @intCast(y[i].v)) - my;
        sum_xy += @divTrunc(dx * dy, @as(i64, Q16.D));
        sum_xx += @divTrunc(dx * dx, @as(i64, Q16.D));
        sum_yy += @divTrunc(dy * dy, @as(i64, Q16.D));
    }

    if (sum_xx == 0 or sum_yy == 0) return Q16.zero();

    const denom_sq: i64 = @divTrunc(sum_xx * sum_yy, @as(i64, Q16.D));
    if (denom_sq <= 0) return Q16.zero();

    const denom = intSqrt(denom_sq * @as(i64, Q16.D));
    if (denom == 0) return Q16.zero();

    const result: i64 = @divTrunc(sum_xy * @as(i64, Q16.D), denom);
    return .{
        .v = @intCast(result),
        .r0 = 0,
    };
}

pub fn statsCovariance(x: []const Q16, y: []const Q16, n: i32) Q16 {
    const count: usize = @intCast(n);
    if (count <= 1) return Q16.zero();

    const mean_x = statsMean(x, n);
    const mean_y = statsMean(y, n);
    const mx: i64 = @intCast(mean_x.v);
    const my: i64 = @intCast(mean_y.v);

    var sum_xy: i64 = 0;
    for (0..count) |i| {
        const dx: i64 = @as(i64, @intCast(x[i].v)) - mx;
        const dy: i64 = @as(i64, @intCast(y[i].v)) - my;
        sum_xy += @divTrunc(dx * dy, @as(i64, Q16.D));
    }

    const ni: i64 = @intCast(n);
    return .{
        .v = @intCast(@divTrunc(sum_xy, ni)),
        .r0 = @intCast(@mod(sum_xy, ni)),
    };
}

pub fn builtinMean(args: *BuiltinArgs) BuiltinResult {
    if (args.input_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    var data: [256]Q16 = undefined;
    const n = @min(args.input_count, 256);
    for (0..@as(usize, @intCast(n))) |i| {
        data[i] = args.input_facts[i].value;
    }
    return dispatch_mod.factResult(statsMean(&data, n));
}

pub fn builtinVariance(args: *BuiltinArgs) BuiltinResult {
    if (args.input_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    var data: [256]Q16 = undefined;
    const n = @min(args.input_count, 256);
    for (0..@as(usize, @intCast(n))) |i| {
        data[i] = args.input_facts[i].value;
    }
    return dispatch_mod.factResult(statsVariance(&data, n));
}

pub fn builtinMedian(args: *BuiltinArgs) BuiltinResult {
    if (args.input_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    var data: [256]Q16 = undefined;
    const n = @min(args.input_count, 256);
    for (0..@as(usize, @intCast(n))) |i| {
        data[i] = args.input_facts[i].value;
    }
    return dispatch_mod.factResult(statsMedian(&data, n));
}

pub fn builtinBayes(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinNormalize(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinHistogram(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinCorrelation(args: *BuiltinArgs) BuiltinResult {
    if (args.input_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const x_val = args.input_facts[0].value;
    const y_val = args.input_facts[1].value;
    var x_arr = [_]Q16{x_val};
    var y_arr = [_]Q16{y_val};
    return dispatch_mod.factResult(statsCorrelation(&x_arr, &y_arr, 1));
}

pub fn builtinCovariance(args: *BuiltinArgs) BuiltinResult {
    if (args.input_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const x_val = args.input_facts[0].value;
    const y_val = args.input_facts[1].value;
    var x_arr = [_]Q16{x_val};
    var y_arr = [_]Q16{y_val};
    return dispatch_mod.factResult(statsCovariance(&x_arr, &y_arr, 1));
}

// ============================================================
// src/builtins/register_linalg.zig
// ============================================================

//NOTE(geoff): This file was not in the original generation, started in turn 21/40

const dispatch_reg = @import("dispatch.zig");
const BuiltinTable = dispatch_reg.BuiltinTable;
const linalg_mod = @import("linalg.zig");
const stats_mod = @import("stats.zig");

pub fn registerLinalgBuiltins(table: *BuiltinTable) void {
    table.register(400, "mat_vec_mul", linalg_mod.builtinMatVecMul, true, 0, .value);
    table.register(401, "transpose", linalg_mod.builtinTranspose, true, 0, .value);
    table.register(402, "gaussian_elim", linalg_mod.builtinGaussianElim, true, 0, .value);
    table.register(403, "inverse", linalg_mod.builtinInverse, true, 0, .value);
    table.register(404, "determinant", linalg_mod.builtinDeterminant, true, 0, .value);
    table.register(405, "gram_schmidt", linalg_mod.builtinGramSchmidt, true, 0, .value);
    table.register(406, "eigenvalues", linalg_mod.builtinEigenvalues, true, 0, .value);
    table.register(407, "svd", linalg_mod.builtinSvd, true, 0, .value);
}

pub fn registerStatsBuiltins(table: *BuiltinTable) void {
    table.register(420, "stats_mean", stats_mod.builtinMean, true, 1, .value);
    table.register(421, "stats_variance", stats_mod.builtinVariance, true, 1, .value);
    table.register(422, "stats_median", stats_mod.builtinMedian, true, 1, .value);
    table.register(423, "stats_bayes", stats_mod.builtinBayes, true, 0, .value);
    table.register(424, "stats_normalize", stats_mod.builtinNormalize, true, 0, .value);
    table.register(425, "stats_histogram", stats_mod.builtinHistogram, true, 0, .value);
    table.register(426, "stats_correlation", stats_mod.builtinCorrelation, true, 2, .value);
    table.register(427, "stats_covariance", stats_mod.builtinCovariance, true, 2, .value);
}

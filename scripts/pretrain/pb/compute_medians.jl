# per-gene nonzero medians over ALL PB samples 

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, Random, Statistics, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
using Preprocess, Args, Config

args = load_pretrain_args()
config = load_config(args["config"], args)
resolve_data_path!(config)   # same data file the finetune scripts load for this data_format

fmt = get(config, "data_format", "tahoe")
data_key = fmt == "lincs" ? "filtered_data" : "df"
data = load(config["data_path"])[data_key]
data_expr = fmt == "lincs" ? data.expr : reduce(hcat, data.expr)
n_genes, n_samples = size(data_expr)
println("$fmt: $n_genes genes × $n_samples samples")
get(config, "subset_ratio", 1.0) < 1.0 && @warn "subset_ratio < 1 is ignored: medians use all samples"

medians = nonzero_medians(data_expr)
n_zero = sum(vec(sum(data_expr .> 0, dims=2)) .== 0)
println("medians over all $n_samples samples: range $(extrema(medians)); $n_zero genes never detected (median set to 1)")

out_path = get(config, "medians_path", "")
out_path = out_path == "" ? default_medians_path(fmt) : out_path
mkpath(dirname(out_path))

jldsave(out_path; medians=medians, samples="all", n_samples=n_samples, data_path=config["data_path"], data_format=fmt,
        scale=(fmt == "lincs" ? "log2 (as stored)" : "log1p CP10k (as stored)"))
println("saved $out_path")

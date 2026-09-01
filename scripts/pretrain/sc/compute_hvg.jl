# SLURM_TIME="1-00:00" cpu_sbatch sc_hvgidx_2c64 julia scripts/pretrain/sc/compute_hvg.jl --config config/local.toml -t etf --hvg_n_shards 2

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, Random, Statistics, Dates

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using Args, Config, LoadSC, ProcessSC

# load_pretrain_args requires -t (modeltype); pass -t etf since HVG is for ETF
args = load_pretrain_args()
config = load_config(args["config"], args)

n_hvg = get(config, "n_hvg", 1024)
n_shards_to_scan = get(config, "hvg_n_shards", 50)
out_path = get(config, "hvg_out", joinpath("data", "hvg_indices.jld2"))

coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])
println("Total shards: $(length(all_shards))")

n_scan = min(n_shards_to_scan, length(all_shards))
scan_shards = shuffle(all_shards)[1:n_scan]
println("Scanning $n_scan shards to compute HVG (n_hvg=$n_hvg)")

# Welford's online algorithm for variance: track count, mean, M2 per gene
gene_count = zeros(Int64, n_coding)
gene_mean = zeros(Float64, n_coding)
gene_m2 = zeros(Float64, n_coding)

dense = Vector{Float32}(undef, n_coding)
total_cells = 0
t_start = now()

for (si, shard_path) in enumerate(scan_shards)
    t_shard = now()
    shard = load_shard_pyarrow(shard_path)

    for ci in 1:shard.n_cells
        cell_to_dense_flat!(dense, shard.genes_flat, shard.offsets, shard.expr_flat,
                            ci, token_to_idx)
        # Welford update for each gene
        for g in 1:n_coding
            x = Float64(dense[g])
            gene_count[g] += 1
            delta = x - gene_mean[g]
            gene_mean[g] += delta / gene_count[g]
            delta2 = x - gene_mean[g]
            gene_m2[g] += delta * delta2
        end
        global total_cells += 1
    end

    elapsed_shard = round((now() - t_shard).value / 1000, digits=1)
    elapsed_total = round((now() - t_start).value / 1000, digits=1)
    avg_per_shard = round(elapsed_total / si, digits=1)
    eta = round(avg_per_shard * (n_scan - si), digits=0)
    println("  [$si/$n_scan] $(basename(shard_path)) — $(shard.n_cells) cells, $(elapsed_shard)s (avg $(avg_per_shard)s/shard, ETA $(eta)s)")
end

total_time = round((now() - t_start).value / 1000, digits=1)
println("Total scan time: $(total_time)s ($(round(total_time/60, digits=1)) min)")

# compute variance
gene_var = zeros(Float64, n_coding)
for g in 1:n_coding
    if gene_count[g] > 1
        gene_var[g] = gene_m2[g] / (gene_count[g] - 1)
    end
end

# select top n_hvg by variance
hvg_idx = sortperm(gene_var, rev=true)[1:n_hvg]
sort!(hvg_idx)  # sorted by gene index for consistent ordering

println("\nResults:")
println("  Scanned $total_cells cells across $n_scan shards")
println("  Top HVG variance range: $(round(gene_var[hvg_idx[1]], digits=4)) — $(round(gene_var[hvg_idx[end]], digits=4))")
println("  Bottom non-HVG variance: $(round(gene_var[sortperm(gene_var, rev=true)[n_hvg+1]], digits=4))")

mkpath(dirname(out_path))
jldsave(out_path; hvg_idx=hvg_idx, gene_var=Float32.(gene_var),
        n_coding=n_coding, n_shards_scanned=n_scan, n_cells_scanned=total_cells)
println("  Saved to $out_path")

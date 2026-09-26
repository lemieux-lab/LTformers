# per-gene nonzero medians of SC log1p CP10k over a seeded random sample of ALL shards 

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, Random, Statistics, Dates

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using Args, Config, LoadSC, ProcessSC

args = load_pretrain_args()
config = load_config(args["config"], args)

n_shards_to_scan = get(config, "medians_n_shards", 100)
out_path = get(config, "sc_medians_path", joinpath("data", "tahoe", "sc_gene_medians.jld2"))
split_seed = get(config, "split_seed", 42)

coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])
n_scan = min(n_shards_to_scan, length(all_shards))
scan_shards = shuffle(MersenneTwister(split_seed), all_shards)[1:n_scan]
println("SC medians: $n_scan of $(length(all_shards)) shards (all splits; sample seed $split_seed)")

const N_BINS = 2000
const V_MAX = 10f0
const BIN_W = V_MAX / N_BINS

# (was a top-level loop over non-const globals: ~12 min/shard; inside a function it's type-stable)
function scan_medians(scan_shards, token_to_idx, n_coding)
    hist = zeros(Int32, N_BINS, n_coding)
    n_nonzero = zeros(Int64, n_coding)
    dense = Vector{Float32}(undef, n_coding)
    total_cells = 0
    t_start = time()
    for (si, shard_path) in enumerate(scan_shards)
        shard = load_shard_pyarrow(shard_path)
        for ci in 1:shard.n_cells
            cell_to_dense_flat!(dense, shard.genes_flat, shard.offsets, shard.expr_flat, ci, token_to_idx)
            @inbounds for g in 1:n_coding
                v = dense[g]
                v > 0f0 || continue
                b = clamp(ceil(Int, v / BIN_W), 1, N_BINS)
                hist[b, g] += Int32(1)
                n_nonzero[g] += 1
            end
        end
        total_cells += shard.n_cells
        println("  [$si/$(length(scan_shards))] $(basename(shard_path)): $(shard.n_cells) cells ($(round(time() - t_start, digits=1))s total)"); flush(stdout)
    end
    # median of the nonzero values per gene from the histogram; never-detected genes get 1 (they rank last anyway)
    medians = ones(Float32, n_coding)
    for g in 1:n_coding
        n = n_nonzero[g]
        n == 0 && continue
        half = n / 2
        cum = 0
        for b in 1:N_BINS
            c = hist[b, g]
            if cum + c >= half
                frac = c == 0 ? 0.0 : (half - cum) / c
                medians[g] = Float32((b - 1 + frac) * BIN_W)
                break
            end
            cum += c
        end
    end
    return medians, n_nonzero, total_cells
end

medians, n_nonzero, total_cells = scan_medians(scan_shards, token_to_idx, n_coding)
println("scanned $total_cells cells; $(sum(n_nonzero .== 0)) genes never detected; median range $(extrema(medians[n_nonzero .> 0]))")

mkpath(dirname(out_path))
jldsave(out_path; medians=medians, n_nonzero=n_nonzero, n_cells_scanned=total_cells,
        scan_shards=basename.(scan_shards), shards="all", sample_seed=split_seed, n_coding=n_coding,
        scale="log1p CP10k (cell_to_dense_flat!)", bin_width=BIN_W)
println("saved $out_path")

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, StatsBase, Statistics, CairoMakie, DataFrames, CSV, PyCall, Random

# per rank: fraction of cells whose gene at that rank has raw count 0 / 1 / 2 / 3 / 4-9 / ≥10,
# and fraction whose count is shared with another detected gene in the cell (order then set only by the gene median)

logln(xs...) = (println(xs...); flush(stdout))

pq = pyimport("pyarrow.parquet")
np = pyimport("numpy")
json_py = pyimport("json")

n_parquets_to_use = parse(Int, get(ENV, "COUNTLV_NPQ", "30"))
res_root = get(ENV, "COUNTLV_ROOT", "results/tahoe/sc")
max_rank = parse(Int, get(ENV, "COUNTLV_MAXRANK", "4000"))
Random.seed!(parse(Int, get(ENV, "COUNTLV_SEED", "42")))
fig_dir = "$res_root/figures/entropies"
data_dir = "$res_root/data/entropies"
mkpath(fig_dir); mkpath(data_dir)

tahoe_data_dir = "data/tahoe/data"
tahoe_meta_dir = "data/tahoe/metadata"

# same coding-gene vocab and medians as entvec.jl
gene_vocab = Dict{Int,String}()
open(joinpath(tahoe_meta_dir, "gene_vocabulary.jsonl")) do f
    for line in eachline(f)
        d = json_py.loads(line)
        gene_vocab[convert(Int, d["token_id"])] = convert(String, d["gene_symbol"])
    end
end
df_coding = CSV.read("data/tahoe/protein-coding_gene.txt", DataFrame; delim='\t')
coding_symbols = Set(df_coding.symbol)
sorted_coding = sort([tid for (tid, sym) in gene_vocab if sym in coding_symbols])
n_coding = length(sorted_coding)
token_to_idx = Dict{Int,Int}(tid => i for (i, tid) in enumerate(sorted_coding))
sc_meds = Float32.(load("data/tahoe/sc_gene_medians.jld2", "medians"))
length(sc_meds) == n_coding || error("medians length $(length(sc_meds)) != $n_coding coding genes")

function read_parquet(path)
    t = pq.read_table(path)
    genes_combined = t.column("genes").combine_chunks()
    expr_combined = t.column("expressions").combine_chunks()
    genes_flat = convert(Vector{Int64}, np.array(genes_combined.values, copy=true))
    offsets = convert(Vector{Int64}, np.array(genes_combined.offsets, copy=true))
    expr_flat = convert(Vector{Float32}, np.array(expr_combined.values, copy=true))
    return (; genes_flat, offsets, expr_flat, n_cells=length(offsets) - 1)
end

function cell_to_raw!(raw, genes_flat, offsets, expr_flat, ci)
    fill!(raw, 0f0)
    s = offsets[ci] + 1; e = offsets[ci + 1]
    expr_flat[s] < 0 && (s += 1)
    for i in s:e
        g = get(token_to_idx, genes_flat[i], 0)
        g == 0 || (raw[g] = expr_flat[i])
    end
    return raw
end

# levels: 1 = 0, 2 = 1, 3 = 2, 4 = 3, 5 = 4-9, 6 = ≥10
level(c) = c == 0 ? 1 : c == 1 ? 2 : c == 2 ? 3 : c == 3 ? 4 : c < 10 ? 5 : 6
level_names = ["0", "1", "2", "3", "4–9", "≥10"]

function chunk!(lv, tied, pqd, cells)
    raw = zeros(Float32, n_coding); normed = similar(raw); ranked = Vector{Int32}(undef, n_coding)
    mult = Dict{Float32,Int}()
    for ci in cells
        cell_to_raw!(raw, pqd.genes_flat, pqd.offsets, pqd.expr_flat, ci)
        total = sum(raw)
        total > 0 || continue
        # ranking as entvec.jl / ProcessSC: log1p CP10k / nonzero median, detected first
        for g in 1:n_coding
            normed[g] = raw[g] > 0 ? log1p(10000f0 * raw[g] / total) / sc_meds[g] : -1f0
        end
        sortperm!(ranked, normed, rev=true)
        empty!(mult)
        for g in 1:n_coding
            raw[g] > 0 && (mult[raw[g]] = get(mult, raw[g], 0) + 1)
        end
        for r in 1:max_rank
            c = raw[ranked[r]]
            lv[level(c), r] += 1
            c > 0 && mult[c] > 1 && (tied[r] += 1)
        end
    end
end

pq_files = sort(filter(f -> endswith(f, ".parquet"), readdir(tahoe_data_dir)))
use_files = pq_files[sort(sample(1:length(pq_files), min(n_parquets_to_use, length(pq_files)), replace=false))]
nt = Threads.nthreads()
lvs = [zeros(Int, 6, max_rank) for _ in 1:nt]
tieds = [zeros(Int, max_rank) for _ in 1:nt]
n_cells = 0
t0 = time()
for (si, f) in enumerate(use_files)
    pqd = read_parquet(joinpath(tahoe_data_dir, f))
    GC.gc()   # PyCall finalizers off the main thread segfault
    parts = collect(Iterators.partition(1:pqd.n_cells, cld(pqd.n_cells, nt)))
    Threads.@threads for c in eachindex(parts)
        chunk!(lvs[c], tieds[c], pqd, parts[c])
    end
    global n_cells += pqd.n_cells
    logln("shard $si / $(length(use_files)), $n_cells cells, $(round(Int, time() - t0)) s")
end
lv = sum(lvs); tied = sum(tieds)
frac = lv ./ sum(lv, dims=1)
tied_frac = tied ./ vec(sum(lv, dims=1))
jldsave("$data_dir/ranked_sc_count_levels.jld2"; counts=lv, frac, tied_frac, level_names, n_cells, n_parquets=length(use_files))

for r in [1, 100, 500, 1000, 1024, 1500, 2000, 3000]
    r > max_rank && continue
    logln("rank $r: ", join(["$(level_names[l])=$(round(frac[l, r], digits=3))" for l in 1:6], "  "),
          "  tied=$(round(tied_frac[r], digits=3))")
end

begin
    fig = Figure(size=(1100, 450))
    cols = reverse(Makie.wong_colors()[1:6])
    ax1 = Axis(fig[1, 1], xlabel="Rank (1 = highest)", ylabel="Fraction of cells",
               title="Raw count of the gene at each rank")
    xs = 1:max_rank
    # stacked from ≥10 (bottom) to 0 (top)
    cum = vcat(zeros(1, max_rank), cumsum(frac[6:-1:1, :], dims=1))
    for (k, l) in enumerate(6:-1:1)
        band!(ax1, xs, cum[k, :], cum[k + 1, :], color=cols[l], label=level_names[l])
    end
    vlines!(ax1, [1024], color=:black, linestyle=:dash)
    xlims!(ax1, 1, max_rank); ylims!(ax1, 0, 1)
    Legend(fig[1, 2], ax1, "Raw count")
    # among cells where the gene at rank r is detected: its count is shared, so its order is set only by the median
    ax2 = Axis(fig[1, 3], xlabel="Rank (1 = highest)", ylabel="Fraction of detected genes", xscale=log10,
               title="Count shared with another detected gene")
    lines!(ax2, xs, tied_frac ./ (1 .- frac[1, :]), color=:black)
    vlines!(ax2, [1024], color=:black, linestyle=:dash)
    xlims!(ax2, 1, max_rank); ylims!(ax2, 0, 1.02)
end
save("$fig_dir/sc_$(length(use_files))_rank_count_levels.png", fig)
logln("saved $fig_dir/sc_$(length(use_files))_rank_count_levels.png")

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, StatsBase, Statistics, CairoMakie, DataFrames, CSV, PyCall, ProgressBars, LinearAlgebra, Random

pq = pyimport("pyarrow.parquet")
np = pyimport("numpy")
json_py = pyimport("json")

save_dir = "results/tahoe/sc/figures/vectors/sc"
mkpath(save_dir)

#######################################################################################################################################

# loading

tahoe_data_dir = "/home/muninn/scratch/kaufmanl/CAP/data/Tahoe-100M/data"
tahoe_meta_dir = "/home/muninn/scratch/kaufmanl/CAP/data/Tahoe-100M/metadata"

# load gene vocabulary (token_id -> gene_symbol)
gene_vocab = Dict{Int,String}()
open(joinpath(tahoe_meta_dir, "gene_vocabulary.jsonl")) do f
    for line in eachline(f)
        d = json_py.loads(line)
        gene_vocab[convert(Int, d["token_id"])] = convert(String, d["gene_symbol"])
    end
end

# load protein-coding gene list and build token filter
df_coding = CSV.read("/home/muninn/scratch/kaufmanl/CAP/data/protein-coding_gene.txt", DataFrame; delim='\t')
coding_symbols = Set(df_coding.symbol)
coding_tokens = Set(tid for (tid, sym) in gene_vocab if sym in coding_symbols)
n_coding = length(coding_tokens)
println("Protein-coding tokens: $n_coding")

# build contiguous index: sorted coding token_id -> 1:n_coding
sorted_coding = sort(collect(coding_tokens))
token_to_idx = Dict{Int,Int}(tid => i for (i, tid) in enumerate(sorted_coding))

# get parquet paths
parquet_files = sort(filter(f -> endswith(f, ".parquet"), readdir(tahoe_data_dir)))
n_parquets = length(parquet_files)
println("Total parquets: $n_parquets")

function read_parquet(path)
    t = pq.read_table(path)
    genes_combined = t.column("genes").combine_chunks()
    expr_combined = t.column("expressions").combine_chunks()
    genes_flat = convert(Vector{Int64}, np.array(genes_combined.values, copy=true))
    offsets = convert(Vector{Int64}, np.array(genes_combined.offsets, copy=true))
    expr_flat = convert(Vector{Float32}, np.array(expr_combined.values, copy=true))
    n_cells = length(offsets) - 1
    return (; genes_flat, offsets, expr_flat, n_cells)
end

function process_cell_to_ranked(genes_flat, offsets, expr_flat, cell_idx, token_to_idx, n_coding)
    vec = cell_to_dense(genes_flat, offsets, expr_flat, cell_idx, token_to_idx, n_coding)
    noise = randn(Float32, n_coding) .* 1f-10
    vec .+= noise
    ranked = Vector{Int32}(undef, n_coding)
    sortperm!(ranked, vec, rev=true)
    return ranked
end

function cell_to_dense(genes_flat, offsets, expr_flat, cell_idx, token_to_idx, n_coding)
    s = offsets[cell_idx] + 1
    e = offsets[cell_idx + 1]
    if expr_flat[s] < 0
        s += 1
    end

    vec = zeros(Float32, n_coding)
    for i in s:e
        ci = get(token_to_idx, genes_flat[i], 0)
        ci == 0 && continue
        vec[ci] = expr_flat[i]
    end

    # log-normalize
    total = sum(vec)
    if total > 0
        for i in 1:n_coding
            if vec[i] > 0
                vec[i] = log1p(10000f0 * vec[i] / total)
            end
        end
    end
    return vec
end

#######################################################################################################################################

### entropy per rank

# n_parquets_to_use = n_parquets
n_parquets_to_use = 5
sampled_parquet_idx = sort(sample(1:n_parquets, min(n_parquets_to_use, n_parquets), replace=false))
sampled_parquet_files = parquet_files[sampled_parquet_idx]

rank_counts = [Dict{Int32,Int}() for _ in 1:n_coding]
rank_zero_counts = zeros(Int, n_coding)   # count of zero-expression genes at each rank
total_cells_processed = 0

println("Entropy: $n_parquets_to_use / $n_parquets parquets")
t0 = time()
for (si, parquet_file) in ProgressBar(enumerate(sampled_parquet_files))
    parquet_path = joinpath(tahoe_data_dir, parquet_file)
    parquet = read_parquet(parquet_path)

    for ci in 1:parquet.n_cells
        dense = cell_to_dense(parquet.genes_flat, parquet.offsets, parquet.expr_flat, ci, token_to_idx, n_coding)
        ranked = process_cell_to_ranked(parquet.genes_flat, parquet.offsets, parquet.expr_flat, ci, token_to_idx, n_coding)

        for (rank_pos, gene_idx) in enumerate(ranked)
            d = rank_counts[rank_pos]
            d[gene_idx] = get(d, gene_idx, 0) + 1
            if dense[gene_idx] == 0.0f0
                rank_zero_counts[rank_pos] += 1
            end
        end
    end
    global total_cells_processed += parquet.n_cells
end
elapsed = time() - t0
println("Total cells processed: $total_cells_processed in $(Int(div(elapsed,3600)))h $(Int(div(elapsed%3600,60)))m $(Int(round(elapsed%60)))s")

# calculate entropy per rank position (only positions with data)
max_populated_rank = findlast(d -> !isempty(d), rank_counts)
sc_entropies = Vector{Float64}(undef, max_populated_rank)
for r in 1:max_populated_rank
    d = rank_counts[r]
    total = sum(values(d))
    if total == 0
        sc_entropies[r] = 0.0
    else
        sc_entropies[r] = -sum(v/total * log2(v/total) for v in values(d))
    end
end

sc_ent_dir = "results/tahoe/sc/data/entropies"
mkpath(sc_ent_dir)
jldsave("$sc_ent_dir/ranked_sc_entropies.jld2"; entropies=sc_entropies)
println("Saved SC entropies to $sc_ent_dir/ranked_sc_entropies.jld2 ($(length(sc_entropies)) ranks)")

# sparsity per rank position (fraction of cells where gene at that rank has zero expression)
sc_sparsities = rank_zero_counts[1:max_populated_rank] ./ total_cells_processed
jldsave("$sc_ent_dir/ranked_sc_sparsities.jld2"; sparsities=sc_sparsities)
println("Saved SC sparsities to $sc_ent_dir/ranked_sc_sparsities.jld2")

begin
    fig_sc_ent = Figure(size=(600, 500))
    ax_sc_ent = Axis(fig_sc_ent[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_sc_ent, 1:max_populated_rank, sc_entropies, alpha=0.5, color=:black)
    display(fig_sc_ent)
end
save("$save_dir/sc_$(n_parquets_to_use)_rank_entropy.png", fig_sc_ent)

### entropy + sparsity overlay (dual y-axis scatter)

begin
    fig_sc_overlay = Figure(size=(600, 500))
    ax_ent = Axis(fig_sc_overlay[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        yaxisposition=:left,
        xtickformat=values -> [string(Int(round(v))) for v in values])
    ax_spar = Axis(fig_sc_overlay[1, 1],
        ylabel="Sparsity (fraction zero)",
        yaxisposition=:right,
        yticklabelcolor=Makie.wong_colors()[1],
        ylabelcolor=Makie.wong_colors()[1])
    hidespines!(ax_spar)
    hidexdecorations!(ax_spar)

    scatter!(ax_ent, 1:max_populated_rank, sc_entropies, alpha=0.5, color=:black, markersize=4, label="Entropy")
    scatter!(ax_spar, 1:max_populated_rank, sc_sparsities, alpha=0.5, color=Makie.wong_colors()[1], markersize=4, label="Sparsity")

    Legend(fig_sc_overlay[0, 1],
        [MarkerElement(color=:black, marker=:circle, markersize=8),
         MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=8)],
        ["Entropy", "Sparsity"],
        orientation=:horizontal, tellwidth=false)

    display(fig_sc_overlay)
end
save("$save_dir/sc_$(n_parquets_to_use)_rank_entropy_sparsity.png", fig_sc_overlay)

# zoomed entropy plots for top-1024 and top-2048
begin
    n_show = min(1024, max_populated_rank)
    fig_1k = Figure(size=(600, 500))
    ax_1k = Axis(fig_1k[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        title="Entropy per rank (top 1024)",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_1k, 1:n_show, sc_entropies[1:n_show], alpha=0.5, color=:black)
    display(fig_1k)
end
save("$save_dir/sc_$(n_parquets_to_use)_rank_entropy_top1024.png", fig_1k)

begin
    n_show = min(2048, max_populated_rank)
    fig_2k = Figure(size=(600, 500))
    ax_2k = Axis(fig_2k[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        title="Entropy per rank (top 2048)",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_2k, 1:n_show, sc_entropies[1:n_show], alpha=0.5, color=:black)
    display(fig_2k)
end
save("$save_dir/sc_$(n_parquets_to_use)_rank_entropy_top2048.png", fig_2k)

#######################################################################################################################################

### mean/std dev expression

sc_gene_count = zeros(Int, n_coding)
sc_gene_mean = zeros(Float64, n_coding)
sc_gene_m2 = zeros(Float64, n_coding)

println("Mean/StdDev: $n_parquets_to_use / $n_parquets parquets")
t0 = time()
for (si, parquet_file) in ProgressBar(enumerate(sampled_parquet_files))
    parquet_path = joinpath(tahoe_data_dir, parquet_file)
    t = pq.read_table(parquet_path)
    genes_combined = t.column("genes").combine_chunks()
    expr_combined = t.column("expressions").combine_chunks()
    gf = convert(Vector{Int64}, np.array(genes_combined.values, copy=true))
    of = convert(Vector{Int64}, np.array(genes_combined.offsets, copy=true))
    ef = convert(Vector{Float32}, np.array(expr_combined.values, copy=true))
    n_cells = length(of) - 1

    for cell_i in 1:n_cells
        s = of[cell_i] + 1
        e = of[cell_i + 1]
        if ef[s] < 0
            s += 1
        end

        # log-normalize this cell
        total = 0f0
        for i in s:e
            total += ef[i]
        end

        for i in s:e
            tid = gf[i]
            gi = get(token_to_idx, tid, 0)
            gi == 0 && continue
            val = total > 0 ? log1p(10000f0 * ef[i] / total) : 0f0

            sc_gene_count[gi] += 1
            delta = val - sc_gene_mean[gi]
            sc_gene_mean[gi] += delta / sc_gene_count[gi]
            delta2 = val - sc_gene_mean[gi]
            sc_gene_m2[gi] += delta * delta2
        end
    end
end
elapsed = time() - t0
println("Mean/StdDev done in $(Int(div(elapsed,3600)))h $(Int(div(elapsed%3600,60)))m $(Int(round(elapsed%60)))s")

sc_gene_std = [sc_gene_m2[i] > 0 && sc_gene_count[i] > 1 ? sqrt(sc_gene_m2[i] / (sc_gene_count[i] - 1)) : 0.0 for i in 1:n_coding]
sc_sorted_by_mean = sortperm(sc_gene_mean, rev=true)

begin
    fig_sc_mean = Figure(size=(600, 400))
    ax_sc_mean = Axis(fig_sc_mean[1, 1],
        xlabel="gene index (sorted by mean expression)",
        ylabel="mean expression level",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_sc_mean, 1:n_coding, sc_gene_mean[sc_sorted_by_mean], alpha=0.5, markersize=5, color=Makie.wong_colors()[2])
    display(fig_sc_mean)
end
save("$save_dir/sc_$(n_parquets_to_use)_gene_exp_mean.png", fig_sc_mean)

begin
    fig_sc_std = Figure(size=(600, 400))
    ax_sc_std = Axis(fig_sc_std[1, 1],
        xlabel="gene index (sorted by mean expression)",
        ylabel="standard deviation",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_sc_std, 1:n_coding, sc_gene_std[sc_sorted_by_mean], alpha=0.5, color=Makie.wong_colors()[3])
    display(fig_sc_std)
end
save("$save_dir/sc_$(n_parquets_to_use)_gene_exp_stddev.png", fig_sc_std, px_per_unit=2)

#######################################################################################################################################

### pairwise vector distances

n_cells_to_sample = 100_000
n_parquets_to_sample = min(n_parquets_to_use, length(parquet_files))
sampled_parquets = sort(sample(1:length(parquet_files), n_parquets_to_sample, replace=false))
cells_per_parquet = cld(n_cells_to_sample, n_parquets_to_sample)

sc_expr = Matrix{Float32}(undef, n_coding, 0)
sc_ranked = Matrix{Int32}(undef, n_coding, 0)
sc_cell_lines = String[]

println("=== Vectors: sampling $n_cells_to_sample cells from $n_parquets_to_sample parquets ===")
t0 = time()
cells_collected = 0
for si in ProgressBar(sampled_parquets)
    cells_collected >= n_cells_to_sample && break
    parquet_path = joinpath(tahoe_data_dir, parquet_files[si])
    parquet = read_parquet(parquet_path)

    # read cell line metadata for this parquet
    parquet_t = pq.read_table(parquet_path, columns=["cell_line_id"])
    parquet_cl = [string(parquet_t.column("cell_line_id").__getitem__(i-1).as_py()) for i in 1:parquet.n_cells]

    n_take = min(cells_per_parquet, parquet.n_cells, n_cells_to_sample - cells_collected)
    cell_idxs = sample(1:parquet.n_cells, n_take, replace=false)

    batch_expr = Matrix{Float32}(undef, n_coding, n_take)
    batch_rank = Matrix{Int32}(undef, n_coding, n_take)
    for (j, ci) in enumerate(cell_idxs)
        batch_expr[:, j] = cell_to_dense(parquet.genes_flat, parquet.offsets, parquet.expr_flat, ci, token_to_idx, n_coding)
        noisy = view(batch_expr, :, j) .+ randn(Float32, n_coding) .* 1f-10
        sortperm!(view(batch_rank, :, j), noisy, rev=true)
    end

    global sc_expr = hcat(sc_expr, batch_expr)
    global sc_ranked = hcat(sc_ranked, batch_rank)
    append!(sc_cell_lines, parquet_cl[cell_idxs])
    global cells_collected += n_take
end
sc_N = size(sc_expr, 2)
elapsed = time() - t0
println("Sampled $sc_N single cells in $(Int(div(elapsed,3600)))h $(Int(div(elapsed%3600,60)))m $(Int(round(elapsed%60)))s")

# pairwise distances (kendall on ~19K-length vectors is slow, so fewer pairs than pseudobulk)
# sc_n_pairs = 1_000_000
sc_n_pairs = 100_000
sc_idx_a = rand(1:sc_N, sc_n_pairs)
sc_idx_b = rand(1:sc_N, sc_n_pairs)

# remove self-pairs
keep = sc_idx_a .!= sc_idx_b
sc_idx_a = sc_idx_a[keep]
sc_idx_b = sc_idx_b[keep]
sc_n_pairs = length(sc_idx_a)
println("After removing self-pairs: $sc_n_pairs pairs")

sc_expr_euclidean = Vector{Float32}(undef, sc_n_pairs)
sc_expr_cosine = Vector{Float32}(undef, sc_n_pairs)

for k in 1:sc_n_pairs
    a = view(sc_expr, :, sc_idx_a[k])
    b = view(sc_expr, :, sc_idx_b[k])
    sc_expr_euclidean[k] = sqrt(sum((a .- b).^2))
    sc_expr_cosine[k] = 1f0 - Float32(dot(a, b) / (norm(a) * norm(b) + 1f-10))
end

sc_rank_kendall = Vector{Float32}(undef, sc_n_pairs)
for k in 1:sc_n_pairs
    σ = view(sc_ranked, :, sc_idx_a[k])
    τ = view(sc_ranked, :, sc_idx_b[k])
    sc_rank_kendall[k] = (1f0 - Float32(corkendall(σ, τ))) / 2f0
end

# save distance vectors
data_vec_dir = "results/tahoe/sc/data/vectors"
mkpath(data_vec_dir)
jldsave("$data_vec_dir/sc_distances_$(sc_n_pairs).jld2";
    euclidean=sc_expr_euclidean, cosine=sc_expr_cosine, kendall=sc_rank_kendall,
    idx_a=sc_idx_a, idx_b=sc_idx_b, n_cells=sc_N, n_parquets=n_parquets_to_use)
println("Saved distance vectors to $data_vec_dir/sc_distances_$(sc_n_pairs).jld2")

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel="kendall tau distance",
        ylabel="euclidean distance")
    rx = (maximum(sc_rank_kendall) - minimum(sc_rank_kendall)) / 100
    ry = (maximum(sc_expr_euclidean) - minimum(sc_expr_euclidean)) / 100
    hb = hexbin!(ax, Float64.(sc_rank_kendall), Float64.(sc_expr_euclidean), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="count (log10)")
    display(fig)
end
save("$save_dir/sc_$(n_parquets_to_use)_euclidean_kendall.png", fig)

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel="kendall tau distance",
        ylabel="cosine distance")
    rx = (maximum(sc_rank_kendall) - minimum(sc_rank_kendall)) / 100
    ry = (maximum(sc_expr_cosine) - minimum(sc_expr_cosine)) / 100
    hb = hexbin!(ax, Float64.(sc_rank_kendall), Float64.(sc_expr_cosine), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="count (log10)")
    display(fig)
end
save("$save_dir/sc_$(n_parquets_to_use)_cosine_kendall.png", fig)

#######################################################################################################################################

### gene overlap analysis — why the blob is diagonal and distances are large

# per-cell library complexity (number of expressed genes)
n_expressed_per_cell = vec(sum(sc_expr .> 0f0, dims=1))
println("\n=== library complexity (genes expressed per cell) ===")
println("  median: $(median(n_expressed_per_cell))  mean: $(round(mean(n_expressed_per_cell), digits=1))  std: $(round(std(n_expressed_per_cell), digits=1))")
println("  min: $(minimum(n_expressed_per_cell))  max: $(maximum(n_expressed_per_cell))  total genes: $n_coding")
println("  median sparsity: $(round(1 - median(n_expressed_per_cell)/n_coding, digits=3))")

# per-pair overlap metrics
sc_n_expressed_a = Vector{Int}(undef, sc_n_pairs)
sc_n_expressed_b = Vector{Int}(undef, sc_n_pairs)
sc_n_shared = Vector{Int}(undef, sc_n_pairs)        # genes nonzero in both
sc_n_union = Vector{Int}(undef, sc_n_pairs)          # genes nonzero in at least one
sc_n_only_a = Vector{Int}(undef, sc_n_pairs)         # nonzero only in a
sc_n_only_b = Vector{Int}(undef, sc_n_pairs)         # nonzero only in b
sc_jaccard = Vector{Float32}(undef, sc_n_pairs)

for k in 1:sc_n_pairs
    a = view(sc_expr, :, sc_idx_a[k])
    b = view(sc_expr, :, sc_idx_b[k])
    nz_a = a .> 0f0
    nz_b = b .> 0f0
    shared = sum(nz_a .& nz_b)
    union = sum(nz_a .| nz_b)
    sc_n_expressed_a[k] = sum(nz_a)
    sc_n_expressed_b[k] = sum(nz_b)
    sc_n_shared[k] = shared
    sc_n_union[k] = union
    sc_n_only_a[k] = sum(nz_a .& .!nz_b)
    sc_n_only_b[k] = sum(.!nz_a .& nz_b)
    sc_jaccard[k] = union > 0 ? Float32(shared / union) : 0f0
end

println("\n=== pairwise gene overlap ===")
println("  jaccard:  median=$(round(median(sc_jaccard), digits=3))  mean=$(round(mean(sc_jaccard), digits=3))")
println("  shared:   median=$(median(sc_n_shared))  mean=$(round(mean(sc_n_shared), digits=1))")
println("  union:    median=$(median(sc_n_union))  mean=$(round(mean(sc_n_union), digits=1))")
println("  only-a:   median=$(median(sc_n_only_a))  mean=$(round(mean(sc_n_only_a), digits=1))")
println("  only-b:   median=$(median(sc_n_only_b))  mean=$(round(mean(sc_n_only_b), digits=1))")

# euclidean contribution from non-overlapping genes (where one is 0 and the other isn't)
sc_euclid_from_mismatch = Vector{Float32}(undef, sc_n_pairs)
sc_euclid_from_shared = Vector{Float32}(undef, sc_n_pairs)
for k in 1:sc_n_pairs
    a = view(sc_expr, :, sc_idx_a[k])
    b = view(sc_expr, :, sc_idx_b[k])
    nz_a = a .> 0f0
    nz_b = b .> 0f0
    shared_mask = nz_a .& nz_b
    mismatch_mask = (nz_a .& .!nz_b) .| (.!nz_a .& nz_b)
    sc_euclid_from_mismatch[k] = sqrt(sum((a[mismatch_mask] .- b[mismatch_mask]).^2))
    sc_euclid_from_shared[k] = sqrt(sum((a[shared_mask] .- b[shared_mask]).^2))
end

frac_from_mismatch = sc_euclid_from_mismatch.^2 ./ (sc_euclid_from_mismatch.^2 .+ sc_euclid_from_shared.^2 .+ 1f-10)
println("\n=== euclidean distance decomposition ===")
println("  fraction of ||a-b||² from non-overlapping genes:")
println("    median=$(round(median(frac_from_mismatch), digits=3))  mean=$(round(mean(frac_from_mismatch), digits=3))")

# save overlap data
jldsave("$data_vec_dir/sc_overlap_$(sc_n_pairs).jld2";
    n_expressed_per_cell=n_expressed_per_cell,
    jaccard=sc_jaccard, n_shared=sc_n_shared, n_union=sc_n_union,
    n_only_a=sc_n_only_a, n_only_b=sc_n_only_b,
    euclid_from_mismatch=sc_euclid_from_mismatch,
    euclid_from_shared=sc_euclid_from_shared,
    frac_from_mismatch=frac_from_mismatch)

### plot 1: library complexity histogram
begin
    fig_lc = Figure(size=(600, 400))
    ax_lc = Axis(fig_lc[1, 1],
        xlabel="Number of expressed genes (per cell)",
        ylabel="Count",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    hist!(ax_lc, Float64.(n_expressed_per_cell), bins=100, color=(:black, 0.6))
    vlines!(ax_lc, [median(n_expressed_per_cell)], color=:red, linewidth=2, linestyle=:dash, label="median=$(Int(median(n_expressed_per_cell)))")
    axislegend(ax_lc, position=:rt)
    display(fig_lc)
end
save("$save_dir/sc_$(n_parquets_to_use)_library_complexity.png", fig_lc)

### plot 2: cosine vs kendall colored by jaccard overlap
begin
    fig_jac = Figure(size=(700, 500))
    ax_jac = Axis(fig_jac[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Cosine distance")
    # subsample for scatter readability
    n_plot = min(20_000, sc_n_pairs)
    plot_idx = sample(1:sc_n_pairs, n_plot, replace=false)
    sc = scatter!(ax_jac,
        Float64.(sc_rank_kendall[plot_idx]),
        Float64.(sc_expr_cosine[plot_idx]),
        color=Float64.(sc_jaccard[plot_idx]),
        colormap=:viridis,
        markersize=3, alpha=0.6)
    Colorbar(fig_jac[1, 2], sc, label="Jaccard overlap")
    display(fig_jac)
end
save("$save_dir/sc_$(n_parquets_to_use)_cosken_by_jaccard.png", fig_jac)

### plot 3: cosine vs kendall colored by fraction of euclidean from non-overlapping genes
begin
    fig_frac = Figure(size=(700, 500))
    ax_frac = Axis(fig_frac[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Cosine distance")
    sc2 = scatter!(ax_frac,
        Float64.(sc_rank_kendall[plot_idx]),
        Float64.(sc_expr_cosine[plot_idx]),
        color=Float64.(frac_from_mismatch[plot_idx]),
        colormap=:inferno,
        markersize=3, alpha=0.6)
    Colorbar(fig_frac[1, 2], sc2, label="Fraction ||Δ||² from\nnon-overlapping genes")
    display(fig_frac)
end
save("$save_dir/sc_$(n_parquets_to_use)_cosken_by_mismatch_frac.png", fig_frac)

### plot 4: jaccard overlap vs cosine distance (direct relationship)
begin
    fig_jc = Figure(size=(600, 500))
    ax_jc = Axis(fig_jc[1, 1],
        xlabel="Jaccard overlap (expressed gene sets)",
        ylabel="Cosine distance")
    scatter!(ax_jc,
        Float64.(sc_jaccard[plot_idx]),
        Float64.(sc_expr_cosine[plot_idx]),
        markersize=2, alpha=0.4, color=:black)
    display(fig_jc)
end
save("$save_dir/sc_$(n_parquets_to_use)_jaccard_vs_cosine.png", fig_jc)

### plot 5: jaccard overlap vs euclidean distance
begin
    fig_je = Figure(size=(600, 500))
    ax_je = Axis(fig_je[1, 1],
        xlabel="Jaccard overlap (expressed gene sets)",
        ylabel="Euclidean distance")
    scatter!(ax_je,
        Float64.(sc_jaccard[plot_idx]),
        Float64.(sc_expr_euclidean[plot_idx]),
        markersize=2, alpha=0.4, color=:black)
    display(fig_je)
end
save("$save_dir/sc_$(n_parquets_to_use)_jaccard_vs_euclidean.png", fig_je)

### plot 6: euclidean decomposition — shared vs mismatch contribution
begin
    fig_decomp = Figure(size=(600, 500))
    ax_decomp = Axis(fig_decomp[1, 1],
        xlabel="Euclidean distance from shared genes",
        ylabel="Euclidean distance from non-overlapping genes")
    sc3 = scatter!(ax_decomp,
        Float64.(sc_euclid_from_shared[plot_idx]),
        Float64.(sc_euclid_from_mismatch[plot_idx]),
        color=Float64.(sc_jaccard[plot_idx]),
        colormap=:viridis,
        markersize=3, alpha=0.6)
    Colorbar(fig_decomp[1, 2], sc3, label="Jaccard overlap")
    # add identity line
    max_val = max(maximum(sc_euclid_from_shared), maximum(sc_euclid_from_mismatch))
    lines!(ax_decomp, [0, max_val], [0, max_val], color=:red, linewidth=1, linestyle=:dash)
    display(fig_decomp)
end
save("$save_dir/sc_$(n_parquets_to_use)_euclid_decomposition.png", fig_decomp)

### plot 7: cosine vs kendall colored by number of shared genes
begin
    fig_nsh = Figure(size=(700, 500))
    ax_nsh = Axis(fig_nsh[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Cosine distance")
    sc4 = scatter!(ax_nsh,
        Float64.(sc_rank_kendall[plot_idx]),
        Float64.(sc_expr_cosine[plot_idx]),
        color=Float64.(sc_n_shared[plot_idx]),
        colormap=:viridis,
        markersize=3, alpha=0.6)
    Colorbar(fig_nsh[1, 2], sc4, label="Shared expressed genes")
    display(fig_nsh)
end
save("$save_dir/sc_$(n_parquets_to_use)_cosken_by_shared.png", fig_nsh)

#######################################################################################################################################

### pairwise vector distances — top-k truncated

for top_k in [1024, 2048]
    println("=== Pairwise distances for top-$top_k ===")
    local t0 = time()

    tk_expr_euclidean = Vector{Float32}(undef, sc_n_pairs)
    tk_expr_cosine = Vector{Float32}(undef, sc_n_pairs)

    for k in 1:sc_n_pairs
        top_genes_a = view(sc_ranked, 1:top_k, sc_idx_a[k])
        top_genes_b = view(sc_ranked, 1:top_k, sc_idx_b[k])
        vals_a = view(sc_expr, :, sc_idx_a[k])
        vals_b = view(sc_expr, :, sc_idx_b[k])

        va = zeros(Float32, n_coding)
        vb = zeros(Float32, n_coding)
        for i in 1:top_k
            va[top_genes_a[i]] = vals_a[top_genes_a[i]]
            vb[top_genes_b[i]] = vals_b[top_genes_b[i]]
        end
        tk_expr_euclidean[k] = sqrt(sum((va .- vb).^2))
        tk_expr_cosine[k] = 1f0 - Float32(dot(va, vb) / (norm(va) * norm(vb) + 1f-10))
    end

    tk_rank_kendall = Vector{Float32}(undef, sc_n_pairs)
    for k in 1:sc_n_pairs
        σ = view(sc_ranked, 1:top_k, sc_idx_a[k])
        τ = view(sc_ranked, 1:top_k, sc_idx_b[k])
        tk_rank_kendall[k] = (1f0 - Float32(corkendall(σ, τ))) / 2f0
    end

    local elapsed = time() - t0
    println("Done in $(Int(div(elapsed,3600)))h $(Int(div(elapsed%3600,60)))m $(Int(round(elapsed%60)))s")

    jldsave("$data_vec_dir/sc_distances_$(sc_n_pairs)_top$(top_k).jld2";
        euclidean=tk_expr_euclidean, cosine=tk_expr_cosine, kendall=tk_rank_kendall,
        idx_a=sc_idx_a, idx_b=sc_idx_b, n_cells=sc_N, n_parquets=n_parquets_to_use, top_k=top_k)
    println("Saved distance vectors to $data_vec_dir/sc_distances_$(sc_n_pairs)_top$(top_k).jld2")

    local fig_euc = begin
        local fig = Figure(size=(600, 500))
        local ax = Axis(fig[1, 1],
            xlabel="kendall tau distance",
            ylabel="euclidean distance",
            title="top-$top_k")
        local rx = (maximum(tk_rank_kendall) - minimum(tk_rank_kendall)) / 100
        local ry = (maximum(tk_expr_euclidean) - minimum(tk_expr_euclidean)) / 100
        local hb = hexbin!(ax, Float64.(tk_rank_kendall), Float64.(tk_expr_euclidean), cellsize=(rx, ry), colorscale=log10)
        Colorbar(fig[1, 2], hb, label="count (log10)")
        display(fig)
        fig
    end
    save("$save_dir/sc_$(n_parquets_to_use)_euclidean_kendall_top$(top_k).png", fig_euc)

    local fig_cos = begin
        local fig = Figure(size=(600, 500))
        local ax = Axis(fig[1, 1],
            xlabel="kendall tau distance",
            ylabel="cosine distance",
            title="top-$top_k")
        local rx = (maximum(tk_rank_kendall) - minimum(tk_rank_kendall)) / 100
        local ry = (maximum(tk_expr_cosine) - minimum(tk_expr_cosine)) / 100
        local hb = hexbin!(ax, Float64.(tk_rank_kendall), Float64.(tk_expr_cosine), cellsize=(rx, ry), colorscale=log10)
        Colorbar(fig[1, 2], hb, label="count (log10)")
        display(fig)
        fig
    end
    save("$save_dir/sc_$(n_parquets_to_use)_cosine_kendall_top$(top_k).png", fig_cos)
end

#######################################################################################################################################

### cell line blob diagnosis (same-CL vs diff-CL pairwise distances)

# results_dir = "out/results"

cl_a = sc_cell_lines[sc_idx_a]
cl_b = sc_cell_lines[sc_idx_b]
same_cl = cl_a .== cl_b

println("\n=== cell line blob diagnosis ===")
println("total pairs: $sc_n_pairs")

low_cos  = sc_expr_cosine .< 0.07
high_cos = sc_expr_cosine .>= 0.07

println("\nlow cosine blob (< 0.07):  n = $(sum(low_cos))")
if sum(low_cos) > 0
    println("  same cell line: $(sum(same_cl .& low_cos)) / $(sum(low_cos)) = $(round(mean(same_cl[low_cos]), digits=3))")
end

println("\nhigh cosine blob (>= 0.07): n = $(sum(high_cos))")
if sum(high_cos) > 0
    println("  same cell line: $(sum(same_cl .& high_cos)) / $(sum(high_cos)) = $(round(mean(same_cl[high_cos]), digits=3))")
end

# break down distances by same vs different cell line
for (label, mask) in [
    ("same cell line",      same_cl),
    ("different cell line", .!same_cl)]
    n = sum(mask)
    n == 0 && continue
    println("\n$label (n=$n):")
    println("  cosine dist:    median=$(round(median(sc_expr_cosine[mask]), digits=4)),  mean=$(round(mean(sc_expr_cosine[mask]), digits=4))")
    println("  euclidean dist: median=$(round(median(sc_expr_euclidean[mask]), digits=4)),  mean=$(round(mean(sc_expr_euclidean[mask]), digits=4))")
    println("  kendall dist:   median=$(round(median(sc_rank_kendall[mask]), digits=4)), mean=$(round(mean(sc_rank_kendall[mask]), digits=4))")
end

# cell line frequency distribution
cl_counts = sort(collect(countmap(sc_cell_lines)), by=x->x[2], rev=true)
println("\n=== cell line distribution (top 10) ===")
for (cl, cnt) in cl_counts[1:min(10, length(cl_counts))]
    println("  $cl: $cnt ($(round(cnt/sc_N*100, digits=1))%)")
end
expected_same_cl = sum((c/sc_N)^2 for (_, c) in cl_counts)
println("expected same-CL rate (random pairs): $(round(expected_same_cl, digits=3))")
println("observed same-CL rate:                $(round(mean(same_cl), digits=3))")

# plot: cosine vs kendall colored by same/diff cell line
begin
    fig_cl = Figure(size=(900, 400))

    ax1 = Axis(fig_cl[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same cell line")
    scatter!(ax1, Float64.(sc_rank_kendall[same_cl]), Float64.(sc_expr_cosine[same_cl]), markersize=1, alpha=0.3, color=:blue)
    scatter!(ax1, Float64.(sc_rank_kendall[.!same_cl]), Float64.(sc_expr_cosine[.!same_cl]), markersize=1, alpha=0.1, color=:gray80)

    ax2 = Axis(fig_cl[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different cell line")
    scatter!(ax2, Float64.(sc_rank_kendall[.!same_cl]), Float64.(sc_expr_cosine[.!same_cl]), markersize=1, alpha=0.3, color=:red)
    scatter!(ax2, Float64.(sc_rank_kendall[same_cl]), Float64.(sc_expr_cosine[same_cl]), markersize=1, alpha=0.1, color=:gray80)

    save("$save_dir/sc_$(n_parquets_to_use)_blob_diagnosis.png", fig_cl)
    display(fig_cl)
end

# plot: euclidean vs kendall colored by same/diff cell line
begin
    fig_cl2 = Figure(size=(900, 400))

    ax1 = Axis(fig_cl2[1, 1], xlabel="kendall tau distance", ylabel="euclidean distance", title="same cell line")
    scatter!(ax1, Float64.(sc_rank_kendall[same_cl]), Float64.(sc_expr_euclidean[same_cl]), markersize=1, alpha=0.3, color=:blue)
    scatter!(ax1, Float64.(sc_rank_kendall[.!same_cl]), Float64.(sc_expr_euclidean[.!same_cl]), markersize=1, alpha=0.1, color=:gray80)

    ax2 = Axis(fig_cl2[1, 2], xlabel="kendall tau distance", ylabel="euclidean distance", title="different cell line")
    scatter!(ax2, Float64.(sc_rank_kendall[.!same_cl]), Float64.(sc_expr_euclidean[.!same_cl]), markersize=1, alpha=0.3, color=:red)
    scatter!(ax2, Float64.(sc_rank_kendall[same_cl]), Float64.(sc_expr_euclidean[same_cl]), markersize=1, alpha=0.1, color=:gray80)

    save("$save_dir/sc_$(n_parquets_to_use)_euclidean_blob_diagnosis.png", fig_cl2)
    display(fig_cl2)
end

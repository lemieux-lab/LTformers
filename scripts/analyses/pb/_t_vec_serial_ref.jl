using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, Statistics, StatsBase, CairoMakie, LinearAlgebra, Random, DataFrames

df = load("data/tahoe/filtered_pseudobulks_alpha_10000.jld2")["df"]
expr = hcat(df.expr...)
# fig_vec_dir = "results/tahoe/pb/figures/vectors"
# fig_var_dir = "results/tahoe/pb/figures/variables"
# data_vec_dir = "results/tahoe/pb/data/vectors"
# knobs (defaults = 100K pairs); everything that depends on the sampled pairs goes into a <pairs> subfolder (100K, 1M, ...)
# VEC_NPAIRS=1000000 julia -t 32 scripts/analyses/pb/t_vec.jl
n_pairs = parse(Int, get(ENV, "VEC_NPAIRS", "100000"))
n_pairs_str = n_pairs >= 1_000_000 ? "$(div(n_pairs, 1_000_000))M" : "$(div(n_pairs, 1_000))K"
res_root = get(ENV, "VEC_ROOT", "results/tahoe/pb")
Random.seed!(parse(Int, get(ENV, "VEC_SEED", "42")))
fig_vec_dir = "$res_root/figures/vectors/$n_pairs_str"
fig_var_dir = "$res_root/figures/variables/$n_pairs_str"
data_vec_dir = "$res_root/data/vectors/$n_pairs_str"
save_prefix = "pb"

n_genes, N = size(expr)
# train nonzero medians, detected first
gene_medians = let p = "data/tahoe/pb_gene_medians.jld2"
    if isfile(p) && length(load(p, "medians")) == size(expr, 1)
        Float32.(load(p, "medians"))
    else
        @warn "no usable medians file $p: computing nonzero medians here"
        Float32[(v = filter(>(0), r); isempty(v) ? 1f0 : median(v)) for r in eachrow(expr)]
    end
end
println("dataset=tahoe  n_genes=$n_genes  N=$N")

mkpath(fig_vec_dir); mkpath(fig_var_dir); mkpath(data_vec_dir)


function rank_genes(expr, medians)
    n, m = size(expr)
    data_ranked = Matrix{Int32}(undef, size(expr))
    normalized_col = Vector{Float32}(undef, n)
    sorted_ind_col = Vector{Int32}(undef, n)
    noise = Vector{Float32}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        sortperm!(sorted_ind_col, normalized_col, rev=true)
        data_ranked[:, j] .= sorted_ind_col
    end
    return data_ranked
end

ranked = rank_genes(expr, gene_medians)


# n_pairs = 100_000

idx_a = rand(1:N, n_pairs)
idx_b = rand(1:N, n_pairs)
for k in 1:n_pairs
    while idx_b[k] == idx_a[k]
        idx_b[k] = rand(1:N)
    end
end

expr_euclid = Vector{Float32}(undef, n_pairs)
expr_cosine = Vector{Float32}(undef, n_pairs)

for k in 1:n_pairs
    a = view(expr, :, idx_a[k])
    b = view(expr, :, idx_b[k])

    expr_euclid[k] = Float32(norm(a .- b))
    expr_cosine[k] = 1f0 - Float32(dot(a, b) / (norm(a) * norm(b)))
end

rank_kendall = Vector{Float32}(undef, n_pairs)
for k in 1:n_pairs
    σ = view(ranked, :, idx_a[k])
    τ = view(ranked, :, idx_b[k])
    # tau-b on x / median, undetected tie
    rank_kendall[k] = (1f0 - Float32(corkendall(Float64.(view(expr, :, idx_a[k]) ./ gene_medians),
                                                Float64.(view(expr, :, idx_b[k]) ./ gene_medians)))) / 2f0
end

# spearman distance (1 - rho) / 2 on x / median; corspearman averages tied ranks, so undetected genes tie
rank_spearman = Vector{Float32}(undef, n_pairs)
for k in 1:n_pairs
    rank_spearman[k] = (1f0 - Float32(corspearman(Float64.(view(expr, :, idx_a[k]) ./ gene_medians),
                                                  Float64.(view(expr, :, idx_b[k]) ./ gene_medians)))) / 2f0
end

# n_pairs_str = n_pairs >= 1_000_000 ? "$(div(n_pairs, 1_000_000))M" : "$(div(n_pairs, 1_000))K"

# cosine/kendall pairs for sep24figs.jl + comparison.jl
# jldsave("$data_vec_dir/cos_ken_$(n_pairs_str)_noself.jld2"; cosine=expr_cosine, kendall=rank_kendall,
#         euclidean=expr_euclid, idx_a=idx_a, idx_b=idx_b)
jldsave("$data_vec_dir/cos_ken_$(n_pairs_str)_noself.jld2"; cosine=expr_cosine, kendall=rank_kendall,
        spearman=rank_spearman, euclidean=expr_euclid, idx_a=idx_a, idx_b=idx_b)


# distances on 1024-gene sets (what the models see), same pairs
#   top1024: each sample's own top-1024 genes (RTF / rlog input). cosine/euclidean on expression with genes outside
#            the sample's top-k set to 0; kendall/spearman on the two top-k lists (Fagin 2003: genes outside a list
#            tie just below it, i.e. score = k+1-position, absent 0; undetected genes are never listed)
#   hvg1024: the 1024 most variable genes (ETF / elog input); all metrics on those genes (rank metrics on x / median)
#   matched: cosine/euclidean on hvg1024 vs kendall/spearman on top1024 (each model family's own input)

hexbin_plot(x, y, xl, yl, title, path) = begin
    fig = Figure(size=(600, 450))
    ax = Axis(fig[1, 1], xlabel=xl, ylabel=yl, title=title)
    rx = (maximum(x) - minimum(x) + 1f-6) / 100; ry = (maximum(y) - minimum(y) + 1f-6) / 100
    hb = hexbin!(ax, Float64.(x), Float64.(y), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="Count (log10)")
    save(path, fig)
    fig
end

top_k = 1024
n_det = vec(sum(expr .> 0f0, dims=1))
function topk_scores(ids_a, ids_b, la, lb, k)
    genes = union(view(ids_a, 1:la), view(ids_b, 1:lb))
    pa = Dict(g => i for (i, g) in enumerate(view(ids_a, 1:la)))
    pb = Dict(g => i for (i, g) in enumerate(view(ids_b, 1:lb)))
    sa = Float64[k + 1 - get(pa, g, k + 1) for g in genes]
    sb = Float64[k + 1 - get(pb, g, k + 1) for g in genes]
    return sa, sb
end

tk_cosine = Vector{Float32}(undef, n_pairs); tk_euclid = Vector{Float32}(undef, n_pairs)
tk_kendall = Vector{Float32}(undef, n_pairs); tk_spearman = Vector{Float32}(undef, n_pairs)
va = zeros(Float32, n_genes); vb = zeros(Float32, n_genes)
for k in 1:n_pairs
    ia, ib = idx_a[k], idx_b[k]
    la, lb = min(top_k, n_det[ia]), min(top_k, n_det[ib])
    fill!(va, 0f0); fill!(vb, 0f0)
    ga = view(ranked, 1:la, ia); gb = view(ranked, 1:lb, ib)
    va[ga] .= view(expr, ga, ia); vb[gb] .= view(expr, gb, ib)
    tk_euclid[k] = Float32(norm(va .- vb))
    tk_cosine[k] = 1f0 - Float32(dot(va, vb) / (norm(va) * norm(vb) + 1f-10))
    sa, sb = topk_scores(view(ranked, :, ia), view(ranked, :, ib), la, lb, top_k)
    tk_kendall[k] = (1f0 - Float32(corkendall(sa, sb))) / 2f0
    tk_spearman[k] = (1f0 - Float32(corspearman(sa, sb))) / 2f0
end
jldsave("$data_vec_dir/pb_distances_top$(top_k)_$(n_pairs_str)_noself.jld2"; cosine=tk_cosine, euclidean=tk_euclid,
        kendall=tk_kendall, spearman=tk_spearman, idx_a=idx_a, idx_b=idx_b, top_k=top_k)

n_hvg = 1024
hvg_idx = sort(sortperm(vec(var(expr, dims=2)), rev=true)[1:n_hvg])
hv_cosine = Vector{Float32}(undef, n_pairs); hv_euclid = Vector{Float32}(undef, n_pairs)
hv_kendall = Vector{Float32}(undef, n_pairs); hv_spearman = Vector{Float32}(undef, n_pairs)
hvg_meds = gene_medians[hvg_idx]
for k in 1:n_pairs
    a = view(expr, hvg_idx, idx_a[k]); b = view(expr, hvg_idx, idx_b[k])
    hv_euclid[k] = Float32(norm(a .- b))
    hv_cosine[k] = 1f0 - Float32(dot(a, b) / (norm(a) * norm(b) + 1f-10))
    ra = Float64.(a ./ hvg_meds); rb = Float64.(b ./ hvg_meds)
    hv_kendall[k] = (1f0 - Float32(corkendall(ra, rb))) / 2f0
    hv_spearman[k] = (1f0 - Float32(corspearman(ra, rb))) / 2f0
end
jldsave("$data_vec_dir/pb_distances_hvg$(n_hvg)_$(n_pairs_str)_noself.jld2"; cosine=hv_cosine, euclidean=hv_euclid,
        kendall=hv_kendall, spearman=hv_spearman, idx_a=idx_a, idx_b=idx_b, hvg_idx=hvg_idx)

# plots: every expression metric vs every rank metric, per gene set (+ model-matched)
gene_sets = [("all", "all $(n_genes) genes", expr_cosine, expr_euclid, rank_kendall, rank_spearman),
             ("top$(top_k)", "top-$(top_k) per sample", tk_cosine, tk_euclid, tk_kendall, tk_spearman),
             ("hvg$(n_hvg)", "HVG-$(n_hvg)", hv_cosine, hv_euclid, hv_kendall, hv_spearman),
             ("matched", "HVG-$(n_hvg) expression vs top-$(top_k) ranks", hv_cosine, hv_euclid, tk_kendall, tk_spearman)]
for (tag, title, cosv, eucv, kenv, spev) in gene_sets
    for (ename, ev) in (("cosine", cosv), ("euclid", eucv)), (rname, rv) in (("kendall", kenv), ("spearman", spev))
        tag == "all" && rname == "kendall" && continue   # existing plots below (pb_{cosine,euclid}_kendall_100K_noself.png)
        hexbin_plot(rv, ev, "$(uppercasefirst(rname)) distance", "$(ename == "cosine" ? "Cosine" : "Euclidean") distance",
                    title, "$fig_vec_dir/$(save_prefix)_$(ename)_$(rname)_$(tag)_$(n_pairs_str)_noself.png")
    end
    println("$tag: spearman(cosine, kendall) = $(round(corspearman(Float64.(cosv), Float64.(kenv)), digits=3)), " *
            "spearman(cosine, spearman) = $(round(corspearman(Float64.(cosv), Float64.(spev)), digits=3))")
end


# gene overlap vs single-cell

# library complexity
pb_n_expressed_per_sample = vec(sum(expr .> 0f0, dims=1))
println("\n=== PB library complexity (genes expressed per pseudobulk) ===")
println("  median: $(median(pb_n_expressed_per_sample))  mean: $(round(mean(pb_n_expressed_per_sample), digits=1))  std: $(round(std(pb_n_expressed_per_sample), digits=1))")
println("  min: $(minimum(pb_n_expressed_per_sample))  max: $(maximum(pb_n_expressed_per_sample))  total genes: $n_genes")
println("  median sparsity: $(round(1 - median(pb_n_expressed_per_sample)/n_genes, digits=3))")
println("  fraction fully dense (all genes expressed): $(round(mean(pb_n_expressed_per_sample .== n_genes), digits=3))")

# pairwise overlap
pb_n_shared = Vector{Int}(undef, n_pairs)
pb_n_union = Vector{Int}(undef, n_pairs)
pb_jaccard = Vector{Float32}(undef, n_pairs)

for k in 1:n_pairs
    a = view(expr, :, idx_a[k])
    b = view(expr, :, idx_b[k])
    nz_a = a .> 0f0
    nz_b = b .> 0f0
    shared = sum(nz_a .& nz_b)
    union = sum(nz_a .| nz_b)
    pb_n_shared[k] = shared
    pb_n_union[k] = union
    pb_jaccard[k] = union > 0 ? Float32(shared / union) : 0f0
end

println("\n=== PB pairwise gene overlap ===")
println("  jaccard:  median=$(round(median(pb_jaccard), digits=3))  mean=$(round(mean(pb_jaccard), digits=3))")
println("  shared:   median=$(median(pb_n_shared))  mean=$(round(mean(pb_n_shared), digits=1))")

# euclidean decomposition
pb_euclid_from_mismatch = Vector{Float32}(undef, n_pairs)
pb_euclid_from_shared = Vector{Float32}(undef, n_pairs)
for k in 1:n_pairs
    a = view(expr, :, idx_a[k])
    b = view(expr, :, idx_b[k])
    nz_a = a .> 0f0
    nz_b = b .> 0f0
    shared_mask = nz_a .& nz_b
    mismatch_mask = (nz_a .& .!nz_b) .| (.!nz_a .& nz_b)
    pb_euclid_from_mismatch[k] = sqrt(sum((a[mismatch_mask] .- b[mismatch_mask]).^2))
    pb_euclid_from_shared[k] = sqrt(sum((a[shared_mask] .- b[shared_mask]).^2))
end

pb_frac_from_mismatch = pb_euclid_from_mismatch.^2 ./ (pb_euclid_from_mismatch.^2 .+ pb_euclid_from_shared.^2 .+ 1f-10)
println("\n=== PB euclidean distance decomposition ===")
println("  fraction of ||a-b||² from non-overlapping genes:")
println("    median=$(round(median(pb_frac_from_mismatch), digits=3))  mean=$(round(mean(pb_frac_from_mismatch), digits=3))")

# save overlap
mkpath(data_vec_dir)
jldsave("$data_vec_dir/pb_overlap_$(n_pairs).jld2";
    n_expressed_per_sample=pb_n_expressed_per_sample,
    jaccard=pb_jaccard, n_shared=pb_n_shared, n_union=pb_n_union,
    euclid_from_mismatch=pb_euclid_from_mismatch,
    euclid_from_shared=pb_euclid_from_shared,
    frac_from_mismatch=pb_frac_from_mismatch)

# PB overlap plots

begin
    fig_lc = Figure(size=(600, 400))
    ax_lc = Axis(fig_lc[1, 1],
        xlabel="Number of expressed genes (per pseudobulk)",
        ylabel="Count",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    hist!(ax_lc, Float64.(pb_n_expressed_per_sample), bins=100, color=(:black, 0.6))
    vlines!(ax_lc, [median(pb_n_expressed_per_sample)], color=:red, linewidth=2, linestyle=:dash, label="median=$(Int(median(pb_n_expressed_per_sample)))")
    axislegend(ax_lc, position=:lt)
    # display(fig_lc)
end
save("$fig_vec_dir/$(save_prefix)_library_complexity.png", fig_lc)

# cosine vs kendall by jaccard
begin
    fig_jac = Figure(size=(700, 500))
    ax_jac = Axis(fig_jac[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Cosine distance")
    n_plot = min(20_000, n_pairs)
    plot_idx_pb = sample(1:n_pairs, n_plot, replace=false)
    sc_jac = scatter!(ax_jac,
        Float64.(rank_kendall[plot_idx_pb]),
        Float64.(expr_cosine[plot_idx_pb]),
        color=Float64.(pb_jaccard[plot_idx_pb]),
        colormap=:viridis,
        markersize=3, alpha=0.6)
    Colorbar(fig_jac[1, 2], sc_jac, label="Jaccard overlap")
    # display(fig_jac)
end
save("$fig_vec_dir/$(save_prefix)_cosken_by_jaccard.png", fig_jac)

# cosine vs kendall by non-overlap euclid fraction
begin
    fig_frac = Figure(size=(700, 500))
    ax_frac = Axis(fig_frac[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Cosine distance")
    sc_frac = scatter!(ax_frac,
        Float64.(rank_kendall[plot_idx_pb]),
        Float64.(expr_cosine[plot_idx_pb]),
        color=Float64.(pb_frac_from_mismatch[plot_idx_pb]),
        colormap=:inferno,
        markersize=3, alpha=0.6)
    Colorbar(fig_frac[1, 2], sc_frac, label="Fraction ||Δ||² from\nnon-overlapping genes")
    # display(fig_frac)
end
save("$fig_vec_dir/$(save_prefix)_cosken_by_mismatch_frac.png", fig_frac)

# jaccard vs cosine
begin
    fig_jc = Figure(size=(600, 500))
    ax_jc = Axis(fig_jc[1, 1],
        xlabel="Jaccard overlap (expressed gene sets)",
        ylabel="Cosine distance")
    scatter!(ax_jc,
        Float64.(pb_jaccard[plot_idx_pb]),
        Float64.(expr_cosine[plot_idx_pb]),
        markersize=2, alpha=0.4, color=:black)
    # display(fig_jc)
end
save("$fig_vec_dir/$(save_prefix)_jaccard_vs_cosine.png", fig_jc)

# euclidean decomposition
begin
    fig_decomp = Figure(size=(600, 500))
    ax_decomp = Axis(fig_decomp[1, 1],
        xlabel="Euclidean distance from shared genes",
        ylabel="Euclidean distance from non-overlapping genes")
    sc_dec = scatter!(ax_decomp,
        Float64.(pb_euclid_from_shared[plot_idx_pb]),
        Float64.(pb_euclid_from_mismatch[plot_idx_pb]),
        color=Float64.(pb_jaccard[plot_idx_pb]),
        colormap=:viridis,
        markersize=3, alpha=0.6)
    Colorbar(fig_decomp[1, 2], sc_dec, label="Jaccard overlap")
    max_val = max(maximum(pb_euclid_from_shared), maximum(pb_euclid_from_mismatch))
    lines!(ax_decomp, [0, max_val], [0, max_val], color=:red, linewidth=1, linestyle=:dash)
    # display(fig_decomp)
end
save("$fig_vec_dir/$(save_prefix)_euclid_decomposition.png", fig_decomp)


# hexbin: euclid/cosine vs kendall

begin
    fig = Figure(size=(600, 400))
    ax = Axis(
        fig[1, 1],
        xlabel="kendall tau distance",
        ylabel="euclidean distance")
    rx = (maximum(rank_kendall) - minimum(rank_kendall)) / 100
    ry = (maximum(expr_euclid) - minimum(expr_euclid)) / 100
    hb = hexbin!(ax, Float64.(rank_kendall), Float64.(expr_euclid), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="count (log10)")
    # display(fig)
end
save("$fig_vec_dir/$(save_prefix)_euclid_kendall_$(n_pairs_str)_noself.png", fig)

begin
    fig = Figure(size=(600, 400))
    ax = Axis(
        fig[1, 1],
        xlabel="kendall tau distance",
        ylabel="cosine distance",
        title="pairwise distances in expression vs. rank vectors")
    rx = (maximum(rank_kendall) - minimum(rank_kendall)) / 100
    ry = (maximum(expr_cosine) - minimum(expr_cosine)) / 100
    hb = hexbin!(ax, Float64.(rank_kendall), Float64.(expr_cosine), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="count (log10)")
    # display(fig)
end
save("$fig_vec_dir/$(save_prefix)_cosine_kendall_$(n_pairs_str)_noself.png", fig)


# Tahoe two-blob diagnosis

drugs_a = df.drug[idx_a]
drugs_b = df.drug[idx_b]
cl_a = df.cell_line[idx_a]
cl_b = df.cell_line[idx_b]

same_drug = drugs_a .== drugs_b
same_cl   = cl_a .== cl_b

println("\n=== blob diagnosis ===")
println("total pairs: $n_pairs")

low_cos  = expr_cosine .< 0.07
high_cos = expr_cosine .>= 0.07

println("\nlow cosine blob (< 0.07):  n = $(sum(low_cos))")
println("  same cell line: $(sum(same_cl .& low_cos)) / $(sum(low_cos)) = $(round(mean(same_cl[low_cos]), digits=3))")
println("  same drug:      $(sum(same_drug .& low_cos)) / $(sum(low_cos)) = $(round(mean(same_drug[low_cos]), digits=3))")

println("\nhigh cosine blob (>= 0.07): n = $(sum(high_cos))")
println("  same cell line: $(sum(same_cl .& high_cos)) / $(sum(high_cos)) = $(round(mean(same_cl[high_cos]), digits=3))")
println("  same drug:      $(sum(same_drug .& high_cos)) / $(sum(high_cos)) = $(round(mean(same_drug[high_cos]), digits=3))")

# by pair type
for (label, mask) in [
    ("same_cl & same_drug",   same_cl .& same_drug),
    ("same_cl & diff_drug",   same_cl .& .!same_drug),
    ("diff_cl & same_drug",  .!same_cl .& same_drug),
    ("diff_cl & diff_drug",  .!same_cl .& .!same_drug)]
    n = sum(mask)
    n == 0 && continue
    println("\n$label (n=$n):")
    println("  cosine dist:    median=$(round(median(expr_cosine[mask]), digits=4)),  mean=$(round(mean(expr_cosine[mask]), digits=4))")
    println("  euclidean dist: median=$(round(median(expr_euclid[mask]), digits=4)),  mean=$(round(mean(expr_euclid[mask]), digits=4))")
    println("  kendall dist:   median=$(round(median(rank_kendall[mask]), digits=4)), mean=$(round(mean(rank_kendall[mask]), digits=4))")
end

# DMSO vs treated
is_dmso_a = drugs_a .== :DMSO
is_dmso_b = drugs_b .== :DMSO
both_dmso   = is_dmso_a .& is_dmso_b
both_trt    = .!is_dmso_a .& .!is_dmso_b
mixed       = (is_dmso_a .& .!is_dmso_b) .| (.!is_dmso_a .& is_dmso_b)

println("\n=== DMSO vs treated ===")
for (label, mask) in [("both DMSO", both_dmso), ("both treated", both_trt), ("mixed (DMSO vs treated)", mixed)]
    n = sum(mask)
    n == 0 && continue
    println("$label (n=$n):")
    println("  cosine dist:    median=$(round(median(expr_cosine[mask]), digits=4))")
    println("  euclidean dist: median=$(round(median(expr_euclid[mask]), digits=4))")
    println("  kendall dist:   median=$(round(median(rank_kendall[mask]), digits=4))")
    println("  fraction in low blob: $(round(mean(low_cos[mask]), digits=3))")
end

# colored by cell line
begin
    fig2 = Figure(size=(900, 400))

    ax1 = Axis(fig2[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same cell line")
    scatter!(ax1, Float64.(rank_kendall[same_cl]), Float64.(expr_cosine[same_cl]), markersize=1, alpha=0.3, color=:blue)
    scatter!(ax1, Float64.(rank_kendall[.!same_cl]), Float64.(expr_cosine[.!same_cl]), markersize=1, alpha=0.1, color=:gray80)

    ax2 = Axis(fig2[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different cell line")
    scatter!(ax2, Float64.(rank_kendall[.!same_cl]), Float64.(expr_cosine[.!same_cl]), markersize=1, alpha=0.3, color=:red)
    scatter!(ax2, Float64.(rank_kendall[same_cl]), Float64.(expr_cosine[same_cl]), markersize=1, alpha=0.1, color=:gray80)

    # display(fig2)
end
save("$fig_var_dir/$(save_prefix)_blob_diagnosis.png", fig2)

# cell line frequencies
cl_counts = sort(collect(countmap(df.cell_line)), by=x->x[2], rev=true)
println("\n=== cell line distribution (top 10) ===")
for (cl, cnt) in cl_counts[1:min(10, length(cl_counts))]
    println("  $cl: $cnt ($(round(cnt/N*100, digits=1))%)")
end
expected_same_cl = sum((c/N)^2 for (_, c) in cl_counts)
println("expected same-CL rate (random pairs): $(round(expected_same_cl, digits=3))")
println("observed same-CL rate:                $(round(mean(same_cl), digits=3))")


# distances by same vs diff cell line

begin
    fig_cl = Figure(size=(1200, 800))

    # cosine
    ax_c1 = Axis(fig_cl[1, 1], xlabel="cosine distance", ylabel="density", title="same cell line")
    hist!(ax_c1, Float64.(expr_cosine[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_c2 = Axis(fig_cl[1, 2], xlabel="cosine distance", ylabel="density", title="different cell line")
    hist!(ax_c2, Float64.(expr_cosine[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    # euclidean
    ax_e1 = Axis(fig_cl[2, 1], xlabel="euclidean distance", ylabel="density")
    hist!(ax_e1, Float64.(expr_euclid[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_e2 = Axis(fig_cl[2, 2], xlabel="euclidean distance", ylabel="density")
    hist!(ax_e2, Float64.(expr_euclid[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    # kendall
    ax_k1 = Axis(fig_cl[3, 1], xlabel="kendall tau distance", ylabel="density")
    hist!(ax_k1, Float64.(rank_kendall[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_k2 = Axis(fig_cl[3, 2], xlabel="kendall tau distance", ylabel="density")
    hist!(ax_k2, Float64.(rank_kendall[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    # display(fig_cl)
end
save("$fig_var_dir/$(save_prefix)_cl_distributions_$(n_pairs_str).png", fig_cl)

# overlaid
begin
    fig_ov = Figure(size=(600, 800))

    ax_co = Axis(fig_ov[1, 1], xlabel="cosine distance", ylabel="density", title="cosine distance")
    hist!(ax_co, Float64.(expr_cosine[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL (n=$(sum(same_cl)))")
    hist!(ax_co, Float64.(expr_cosine[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL (n=$(sum(.!same_cl)))")
    axislegend(ax_co, position=:rt)

    ax_eu = Axis(fig_ov[2, 1], xlabel="euclidean distance", ylabel="density", title="euclidean distance")
    hist!(ax_eu, Float64.(expr_euclid[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL")
    hist!(ax_eu, Float64.(expr_euclid[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL")
    axislegend(ax_eu, position=:rt)

    ax_ke = Axis(fig_ov[3, 1], xlabel="kendall tau distance", ylabel="density", title="kendall tau distance")
    hist!(ax_ke, Float64.(rank_kendall[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL")
    hist!(ax_ke, Float64.(rank_kendall[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL")
    axislegend(ax_ke, position=:rt)

    # display(fig_ov)
end
save("$fig_var_dir/$(save_prefix)_cl_overlay_$(n_pairs_str).png", fig_ov)

# density vs kendall by cell line
begin
    fig_clk = Figure(size=(700, 450))
    ax_clk = Axis(fig_clk[1, 1],
        xlabel="Kendall tau distance",
        ylabel="Density",
        title="Same vs. different cell line (Tahoe pseudobulk)")
    hist!(ax_clk, Float64.(rank_kendall[same_cl]), bins=120, normalization=:pdf,
          color=(:blue, 0.4), label="same CL (n=$(sum(same_cl)))")
    hist!(ax_clk, Float64.(rank_kendall[.!same_cl]), bins=120, normalization=:pdf,
          color=(:red, 0.4), label="diff CL (n=$(sum(.!same_cl)))")
    axislegend(ax_clk, position=:lt)
    # display(fig_clk)
end
save("$fig_var_dir/$(save_prefix)_cl_kendall_density_$(n_pairs_str).png", fig_clk)


# more confounders

plate_a = df.plate[idx_a]
plate_b = df.plate[idx_b]
same_plate = plate_a .== plate_b

dose_a = df.dose[idx_a]
dose_b = df.dose[idx_b]
same_dose = dose_a .== dose_b

println("\n=== plate confounder ===")
println("same-plate base rate: $(round(mean(same_plate), digits=3))")
println("same-plate in low blob:  $(round(mean(same_plate[low_cos]), digits=3))")
println("same-plate in high blob: $(round(mean(same_plate[high_cos]), digits=3))")

println("\n=== dose confounder ===")
println("same-dose base rate: $(round(mean(same_dose), digits=3))")
println("same-dose in low blob:  $(round(mean(same_dose[low_cos]), digits=3))")
println("same-dose in high blob: $(round(mean(same_dose[high_cos]), digits=3))")

println("\n=== same drug & same dose ===")
same_drug_dose = same_drug .& same_dose
println("same-drug-dose base rate: $(round(mean(same_drug_dose), digits=4))")
println("same-drug-dose in low blob:  $(round(mean(same_drug_dose[low_cos]), digits=4))")
println("same-drug-dose in high blob: $(round(mean(same_drug_dose[high_cos]), digits=4))")

# diff-CL pairs in low-cosine blob
diff_cl_low = .!same_cl .& low_cos
println("\n=== diff-CL pairs in low cosine blob (n=$(sum(diff_cl_low))) ===")
if sum(diff_cl_low) > 0
    cl_pairs_low = countmap(collect(zip(
        min.(cl_a[diff_cl_low], cl_b[diff_cl_low]),
        max.(cl_a[diff_cl_low], cl_b[diff_cl_low]))))
    sorted_pairs = sort(collect(cl_pairs_low), by=x->x[2], rev=true)
    for (pair, cnt) in sorted_pairs[1:min(20, length(sorted_pairs))]
        println("  $(pair[1]) — $(pair[2]): $cnt")
    end
end

# colored by confounder
# plate
begin
    fig_plate = Figure(size=(900, 400))
    ax_p1 = Axis(fig_plate[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same plate")
    scatter!(ax_p1, Float64.(rank_kendall[same_plate]), Float64.(expr_cosine[same_plate]), markersize=1, alpha=0.3, color=:purple)
    scatter!(ax_p1, Float64.(rank_kendall[.!same_plate]), Float64.(expr_cosine[.!same_plate]), markersize=1, alpha=0.05, color=:gray80)
    ax_p2 = Axis(fig_plate[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different plate")
    scatter!(ax_p2, Float64.(rank_kendall[.!same_plate]), Float64.(expr_cosine[.!same_plate]), markersize=1, alpha=0.3, color=:darkorange)
    scatter!(ax_p2, Float64.(rank_kendall[same_plate]), Float64.(expr_cosine[same_plate]), markersize=1, alpha=0.05, color=:gray80)
    # display(fig_plate)
end
save("$fig_var_dir/$(save_prefix)_plate_diagnosis.png", fig_plate)

# dose
begin
    fig_dose = Figure(size=(900, 400))
    ax_d1 = Axis(fig_dose[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same dose")
    scatter!(ax_d1, Float64.(rank_kendall[same_dose]), Float64.(expr_cosine[same_dose]), markersize=1, alpha=0.3, color=:green)
    scatter!(ax_d1, Float64.(rank_kendall[.!same_dose]), Float64.(expr_cosine[.!same_dose]), markersize=1, alpha=0.05, color=:gray80)
    ax_d2 = Axis(fig_dose[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different dose")
    scatter!(ax_d2, Float64.(rank_kendall[.!same_dose]), Float64.(expr_cosine[.!same_dose]), markersize=1, alpha=0.3, color=:brown)
    scatter!(ax_d2, Float64.(rank_kendall[same_dose]), Float64.(expr_cosine[same_dose]), markersize=1, alpha=0.05, color=:gray80)
    # display(fig_dose)
end
save("$fig_var_dir/$(save_prefix)_dose_diagnosis.png", fig_dose)

# confounder cosine histograms
begin
    fig_conf = Figure(size=(700, 900))

    ax_cf1 = Axis(fig_conf[1, 1], xlabel="cosine distance", ylabel="density", title="cell line")
    hist!(ax_cf1, Float64.(expr_cosine[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL (n=$(sum(same_cl)))")
    hist!(ax_cf1, Float64.(expr_cosine[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL (n=$(sum(.!same_cl)))")
    axislegend(ax_cf1, position=:rt)

    ax_cf2 = Axis(fig_conf[2, 1], xlabel="cosine distance", ylabel="density", title="plate")
    hist!(ax_cf2, Float64.(expr_cosine[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_cf2, Float64.(expr_cosine[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_cf2, position=:rt)

    ax_cf3 = Axis(fig_conf[3, 1], xlabel="cosine distance", ylabel="density", title="dose")
    hist!(ax_cf3, Float64.(expr_cosine[same_dose]), bins=100, normalization=:pdf, color=(:green, 0.4), label="same dose (n=$(sum(same_dose)))")
    hist!(ax_cf3, Float64.(expr_cosine[.!same_dose]), bins=100, normalization=:pdf, color=(:brown, 0.4), label="diff dose (n=$(sum(.!same_dose)))")
    axislegend(ax_cf3, position=:rt)

    ax_cf4 = Axis(fig_conf[4, 1], xlabel="cosine distance", ylabel="density", title="drug")
    hist!(ax_cf4, Float64.(expr_cosine[same_drug]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same drug (n=$(sum(same_drug)))")
    hist!(ax_cf4, Float64.(expr_cosine[.!same_drug]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff drug (n=$(sum(.!same_drug)))")
    axislegend(ax_cf4, position=:rt)

    # display(fig_conf)
end
save("$fig_var_dir/$(save_prefix)_confounder_cosine_$(n_pairs_str).png", fig_conf)

# confounder euclid histograms
begin
    fig_conf_e = Figure(size=(700, 900))

    ax_ce1 = Axis(fig_conf_e[1, 1], xlabel="euclidean distance", ylabel="density", title="cell line")
    hist!(ax_ce1, Float64.(expr_euclid[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL (n=$(sum(same_cl)))")
    hist!(ax_ce1, Float64.(expr_euclid[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL (n=$(sum(.!same_cl)))")
    axislegend(ax_ce1, position=:lt)

    ax_ce2 = Axis(fig_conf_e[2, 1], xlabel="euclidean distance", ylabel="density", title="plate")
    hist!(ax_ce2, Float64.(expr_euclid[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_ce2, Float64.(expr_euclid[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_ce2, position=:lt)

    ax_ce3 = Axis(fig_conf_e[3, 1], xlabel="euclidean distance", ylabel="density", title="dose")
    hist!(ax_ce3, Float64.(expr_euclid[same_dose]), bins=100, normalization=:pdf, color=(:green, 0.4), label="same dose (n=$(sum(same_dose)))")
    hist!(ax_ce3, Float64.(expr_euclid[.!same_dose]), bins=100, normalization=:pdf, color=(:brown, 0.4), label="diff dose (n=$(sum(.!same_dose)))")
    axislegend(ax_ce3, position=:lt)

    ax_ce4 = Axis(fig_conf_e[4, 1], xlabel="euclidean distance", ylabel="density", title="drug")
    hist!(ax_ce4, Float64.(expr_euclid[same_drug]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same drug (n=$(sum(same_drug)))")
    hist!(ax_ce4, Float64.(expr_euclid[.!same_drug]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff drug (n=$(sum(.!same_drug)))")
    axislegend(ax_ce4, position=:lt)

    # display(fig_conf_e)
end
save("$fig_var_dir/$(save_prefix)_confounder_euclidean_$(n_pairs_str).png", fig_conf_e)

# confounder kendall histograms
begin
    fig_conf_k = Figure(size=(700, 900))

    ax_ck1 = Axis(fig_conf_k[1, 1], xlabel="kendall tau distance", ylabel="density", title="cell line")
    hist!(ax_ck1, Float64.(rank_kendall[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.4), label="same CL (n=$(sum(same_cl)))")
    hist!(ax_ck1, Float64.(rank_kendall[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.4), label="diff CL (n=$(sum(.!same_cl)))")
    axislegend(ax_ck1, position=:lt)

    ax_ck2 = Axis(fig_conf_k[2, 1], xlabel="kendall tau distance", ylabel="density", title="plate")
    hist!(ax_ck2, Float64.(rank_kendall[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_ck2, Float64.(rank_kendall[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_ck2, position=:lt)

    ax_ck3 = Axis(fig_conf_k[3, 1], xlabel="kendall tau distance", ylabel="density", title="dose")
    hist!(ax_ck3, Float64.(rank_kendall[same_dose]), bins=100, normalization=:pdf, color=(:green, 0.4), label="same dose (n=$(sum(same_dose)))")
    hist!(ax_ck3, Float64.(rank_kendall[.!same_dose]), bins=100, normalization=:pdf, color=(:brown, 0.4), label="diff dose (n=$(sum(.!same_dose)))")
    axislegend(ax_ck3, position=:lt)

    ax_ck4 = Axis(fig_conf_k[4, 1], xlabel="kendall tau distance", ylabel="density", title="drug")
    hist!(ax_ck4, Float64.(rank_kendall[same_drug]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same drug (n=$(sum(same_drug)))")
    hist!(ax_ck4, Float64.(rank_kendall[.!same_drug]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff drug (n=$(sum(.!same_drug)))")
    axislegend(ax_ck4, position=:lt)

    # display(fig_conf_k)
end
save("$fig_var_dir/$(save_prefix)_confounder_kendall_$(n_pairs_str).png", fig_conf_k)

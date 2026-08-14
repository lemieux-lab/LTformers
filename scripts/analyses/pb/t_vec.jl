using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, Statistics, StatsBase, CairoMakie, LinearAlgebra, Random, DataFrames

df = load("/home/muninn/scratch/kaufmanl/CAP/results/tahoe/pseudobulks/filtered_pseudobulks_alpha_10000.jld2")["df"]
expr = hcat(df.expr...)
fig_vec_dir = "results/tahoe/pb/figures/vectors"
fig_var_dir = "results/tahoe/pb/figures/variables"
data_vec_dir = "results/tahoe/pb/data/vectors"
save_prefix = "pb"

n_genes, N = size(expr)
gene_medians = vec(median(expr, dims=2)) .+ 1f-10
println("dataset=tahoe  n_genes=$n_genes  N=$N")

mkpath(fig_vec_dir); mkpath(fig_var_dir); mkpath(data_vec_dir)

#######################################################################################################################################

function rank_genes(expr, medians)
    n, m = size(expr)
    data_ranked = Matrix{Int32}(undef, size(expr))
    normalized_col = Vector{Float32}(undef, n)
    sorted_ind_col = Vector{Int32}(undef, n)
    noise = Vector{Float32}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        randn!(noise)
        @. normalized_col += noise * 1f-10 # only doing the noise here bc thats what we doing in the single-cell
        sortperm!(sorted_ind_col, normalized_col, rev=true)
        data_ranked[:, j] .= sorted_ind_col
    end
    return data_ranked
end

ranked = rank_genes(expr, gene_medians)

#######################################################################################################################################

n_pairs = 100_000

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
    rank_kendall[k] = (1f0 - Float32(corkendall(σ, τ))) / 2f0
end

n_pairs_str = n_pairs >= 1_000_000 ? "$(div(n_pairs, 1_000_000))M" : "$(div(n_pairs, 1_000))K"

#######################################################################################################################################

### gene overlap analysis — contrast with single-cell

# per-sample library complexity
pb_n_expressed_per_sample = vec(sum(expr .> 0f0, dims=1))
println("\n=== PB library complexity (genes expressed per pseudobulk) ===")
println("  median: $(median(pb_n_expressed_per_sample))  mean: $(round(mean(pb_n_expressed_per_sample), digits=1))  std: $(round(std(pb_n_expressed_per_sample), digits=1))")
println("  min: $(minimum(pb_n_expressed_per_sample))  max: $(maximum(pb_n_expressed_per_sample))  total genes: $n_genes")
println("  median sparsity: $(round(1 - median(pb_n_expressed_per_sample)/n_genes, digits=3))")
println("  fraction fully dense (all genes expressed): $(round(mean(pb_n_expressed_per_sample .== n_genes), digits=3))")

# per-pair overlap metrics
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

# save overlap data
mkpath(data_vec_dir)
jldsave("$data_vec_dir/pb_overlap_$(n_pairs).jld2";
    n_expressed_per_sample=pb_n_expressed_per_sample,
    jaccard=pb_jaccard, n_shared=pb_n_shared, n_union=pb_n_union,
    euclid_from_mismatch=pb_euclid_from_mismatch,
    euclid_from_shared=pb_euclid_from_shared,
    frac_from_mismatch=pb_frac_from_mismatch)

### PB overlap plots

begin
    fig_lc = Figure(size=(600, 400))
    ax_lc = Axis(fig_lc[1, 1],
        xlabel="Number of expressed genes (per pseudobulk)",
        ylabel="Count",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    hist!(ax_lc, Float64.(pb_n_expressed_per_sample), bins=100, color=(:black, 0.6))
    vlines!(ax_lc, [median(pb_n_expressed_per_sample)], color=:red, linewidth=2, linestyle=:dash, label="median=$(Int(median(pb_n_expressed_per_sample)))")
    axislegend(ax_lc, position=:lt)
    display(fig_lc)
end
save("$fig_vec_dir/$(save_prefix)_library_complexity.png", fig_lc)

# cosine vs kendall colored by jaccard
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
    display(fig_jac)
end
save("$fig_vec_dir/$(save_prefix)_cosken_by_jaccard.png", fig_jac)

# cosine vs kendall colored by fraction of euclidean from non-overlapping genes
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
    display(fig_frac)
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
    display(fig_jc)
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
    display(fig_decomp)
end
save("$fig_vec_dir/$(save_prefix)_euclid_decomposition.png", fig_decomp)

#######################################################################################################################################

# hexbin plots: euclidean vs kendall, cosine vs kendall

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
    display(fig)
end
# save("$fig_vec_dir/$(save_prefix)_euclid_kendall_$(n_pairs_str)_noself.png", fig)

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
    display(fig)
end

#######################################################################################################################################

### Tahoe: diagnose two-blob structure using drug / cell_line / plate / dose

drugs_a = df.drug[idx_a]
drugs_b = df.drug[idx_b]
cl_a = df.cell_line[idx_a]
cl_b = df.cell_line[idx_b]

same_drug = drugs_a .== drugs_b
same_cl   = cl_a .== cl_b

println("\n=== blob diagnosis ===")
println("total pairs: $n_pairs")

low_cos  = expr_cosine .< 0.07  # threshold between the two blobs
high_cos = expr_cosine .>= 0.07

println("\nlow cosine blob (< 0.07):  n = $(sum(low_cos))")
println("  same cell line: $(sum(same_cl .& low_cos)) / $(sum(low_cos)) = $(round(mean(same_cl[low_cos]), digits=3))")
println("  same drug:      $(sum(same_drug .& low_cos)) / $(sum(low_cos)) = $(round(mean(same_drug[low_cos]), digits=3))")

println("\nhigh cosine blob (>= 0.07): n = $(sum(high_cos))")
println("  same cell line: $(sum(same_cl .& high_cos)) / $(sum(high_cos)) = $(round(mean(same_cl[high_cos]), digits=3))")
println("  same drug:      $(sum(same_drug .& high_cos)) / $(sum(high_cos)) = $(round(mean(same_drug[high_cos]), digits=3))")

# break down by pair type
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

# DMSO control vs treated
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

# scatter plots colored by cell line
begin
    fig2 = Figure(size=(900, 400))

    ax1 = Axis(fig2[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same cell line")
    scatter!(ax1, Float64.(rank_kendall[same_cl]), Float64.(expr_cosine[same_cl]), markersize=1, alpha=0.3, color=:blue)
    scatter!(ax1, Float64.(rank_kendall[.!same_cl]), Float64.(expr_cosine[.!same_cl]), markersize=1, alpha=0.1, color=:gray80)

    ax2 = Axis(fig2[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different cell line")
    scatter!(ax2, Float64.(rank_kendall[.!same_cl]), Float64.(expr_cosine[.!same_cl]), markersize=1, alpha=0.3, color=:red)
    scatter!(ax2, Float64.(rank_kendall[same_cl]), Float64.(expr_cosine[same_cl]), markersize=1, alpha=0.1, color=:gray80)

    display(fig2)
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

###################################################################################################################################

# distribution of each distance metric split by same vs different cell line

begin
    fig_cl = Figure(size=(1200, 800))

    # cosine distance
    ax_c1 = Axis(fig_cl[1, 1], xlabel="cosine distance", ylabel="density", title="same cell line")
    hist!(ax_c1, Float64.(expr_cosine[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_c2 = Axis(fig_cl[1, 2], xlabel="cosine distance", ylabel="density", title="different cell line")
    hist!(ax_c2, Float64.(expr_cosine[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    # euclidean distance
    ax_e1 = Axis(fig_cl[2, 1], xlabel="euclidean distance", ylabel="density")
    hist!(ax_e1, Float64.(expr_euclid[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_e2 = Axis(fig_cl[2, 2], xlabel="euclidean distance", ylabel="density")
    hist!(ax_e2, Float64.(expr_euclid[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    # kendall tau distance
    ax_k1 = Axis(fig_cl[3, 1], xlabel="kendall tau distance", ylabel="density")
    hist!(ax_k1, Float64.(rank_kendall[same_cl]), bins=100, normalization=:pdf, color=(:blue, 0.5))
    ax_k2 = Axis(fig_cl[3, 2], xlabel="kendall tau distance", ylabel="density")
    hist!(ax_k2, Float64.(rank_kendall[.!same_cl]), bins=100, normalization=:pdf, color=(:red, 0.5))

    display(fig_cl)
end
save("$fig_var_dir/$(save_prefix)_cl_distributions_$(n_pairs_str).png", fig_cl)

# overlaid version
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

    display(fig_ov)
end
save("$fig_var_dir/$(save_prefix)_cl_overlay_$(n_pairs_str).png", fig_ov)

# standalone density vs kendall tau: same cell line vs different cell line
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
    display(fig_clk)
end
save("$fig_var_dir/$(save_prefix)_cl_kendall_density_$(n_pairs_str).png", fig_clk)

###################################################################################################################################

# additional confounder checks: plate, dose, same-drug-same-dose, diff-CL pair identities

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

# which diff-CL pairs land in the low-cosine blob?
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

# scatter plots colored by each confounder
# plate
begin
    fig_plate = Figure(size=(900, 400))
    ax_p1 = Axis(fig_plate[1, 1], xlabel="kendall tau distance", ylabel="cosine distance", title="same plate")
    scatter!(ax_p1, Float64.(rank_kendall[same_plate]), Float64.(expr_cosine[same_plate]), markersize=1, alpha=0.3, color=:purple)
    scatter!(ax_p1, Float64.(rank_kendall[.!same_plate]), Float64.(expr_cosine[.!same_plate]), markersize=1, alpha=0.05, color=:gray80)
    ax_p2 = Axis(fig_plate[1, 2], xlabel="kendall tau distance", ylabel="cosine distance", title="different plate")
    scatter!(ax_p2, Float64.(rank_kendall[.!same_plate]), Float64.(expr_cosine[.!same_plate]), markersize=1, alpha=0.3, color=:darkorange)
    scatter!(ax_p2, Float64.(rank_kendall[same_plate]), Float64.(expr_cosine[same_plate]), markersize=1, alpha=0.05, color=:gray80)
    display(fig_plate)
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
    display(fig_dose)
end
save("$fig_var_dir/$(save_prefix)_dose_diagnosis.png", fig_dose)

# combined confounder summary: overlaid cosine histograms
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

    display(fig_conf)
end
save("$fig_var_dir/$(save_prefix)_confounder_cosine_$(n_pairs_str).png", fig_conf)

# euclidean version of confounder summary
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

    display(fig_conf_e)
end
save("$fig_var_dir/$(save_prefix)_confounder_euclidean_$(n_pairs_str).png", fig_conf_e)

# kendall tau version of confounder summary
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

    display(fig_conf_k)
end
save("$fig_var_dir/$(save_prefix)_confounder_kendall_$(n_pairs_str).png", fig_conf_k)

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, Statistics, StatsBase, CairoMakie, LinearAlgebra, Random, DataFrames

# data = "mcf7_10um_24h.jld2"
data = "data_expr.jld2"
expr = load("data/lincs/$data")["data_expr"]
fig_vec_cosine_dir = "results/lincs/figures/vectors/cosine"
fig_vec_euclid_dir = "results/lincs/figures/vectors/euclid"
fig_var_dir = "results/lincs/figures/variables"
data_vec_cosine_dir = "results/lincs/data/vectors/cosine"
data_vec_euclid_dir = "results/lincs/data/vectors/euclid"
save_prefix = "lincs"

n_genes, N = size(expr)
gene_medians = vec(median(expr, dims=2)) .+ 1f-10
println("dataset=lincs  n_genes=$n_genes  N=$N")

mkpath(fig_vec_cosine_dir); mkpath(fig_vec_euclid_dir); mkpath(fig_var_dir)
mkpath(data_vec_cosine_dir); mkpath(data_vec_euclid_dir)

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
# save("$fig_vec_euclid_dir/euclid_kendall_$(n_pairs_str)_noself.png", fig)

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

### LINCS: diagnose upper blob using det_plate / pert_id / sample_id

mfc = load("data/lincs/lincs_trt_inst.jld2")["mfc"]
pert_id = load("data/lincs/lincs_trt_inst.jld2")["pert_id"]

inst = load("data/lincs/lincs_trt_data.jld2")["filtered_data"].inst
sample_ids = inst.sample_id
det_plates = inst.det_plate
cmap_names = inst.cmap_name

# frequency in upper tail
tail_threshold = 0.10f0
tail_mask = expr_cosine .> tail_threshold
tail_indices = findall(tail_mask)

function tail_freq(ids, idx_a, idx_b, tail_indices)
    countmap(vcat(
        [ids[idx_a[k]] for k in tail_indices],
        [ids[idx_b[k]] for k in tail_indices]))
end

# compounds (pert_id + cmap_name)
tail_perts = tail_freq(pert_id, idx_a, idx_b, tail_indices)
tail_perts_sorted = sort(collect(tail_perts), by=x->-x[2])
overall_perts = countmap(pert_id)
pid_to_name = Dict(zip(pert_id, cmap_names))
for (pid, count) in tail_perts_sorted[1:min(20, length(tail_perts_sorted))]
    overall_count = get(overall_perts, pid, 0)
    overall_frac = round(100 * overall_count / length(pert_id), digits=2)
    name = get(pid_to_name, pid, "?")
    println("$pid ($name): $count tail appearances (overall: $overall_count samples, $overall_frac%)")
end

# individual samples (sample_id)
tail_samples = tail_freq(sample_ids, idx_a, idx_b, tail_indices)
tail_samples_sorted = sort(collect(tail_samples), by=x->-x[2])
for (sid, count) in tail_samples_sorted[1:min(20, length(tail_samples_sorted))]
    println("$sid: $count")
end

# plates (det_plate)
tail_plates = tail_freq(det_plates, idx_a, idx_b, tail_indices)
tail_plates_sorted = sort(collect(tail_plates), by=x->-x[2])
overall_plates = countmap(det_plates)
for (plate, count) in tail_plates_sorted[1:min(20, length(tail_plates_sorted))]
    overall_count = get(overall_plates, plate, 0)
    overall_frac = round(100 * overall_count / length(det_plates), digits=2)
    println("$plate: $count tail appearances (overall: $overall_count samples, $overall_frac%)")
end

#=
from the above, we see that REP.A010_JURKAT_24H_X3_B32 is the main cause of the upper blob;
this is part of det_plate
so we can remove it and redo:
=#

bad_plate = Symbol("REP.A010_JURKAT_24H_X3_B32")
clean_mask = [(det_plates[idx_a[k]] != bad_plate) && (det_plates[idx_b[k]] != bad_plate) for k in 1:n_pairs]
println("pairs before: $n_pairs, after: $(sum(clean_mask))")

clean_cosine = expr_cosine[clean_mask]
clean_kendall = rank_kendall[clean_mask]
# clean_rmse = expr_rmse[clean_mask]
clean_euclid = expr_euclid[clean_mask]

jldsave("$data_vec_euclid_dir/euc_ken_cleaned_$(n_pairs_str)_noself.jld2"; euclid=clean_euclid, kendall=clean_kendall)
jldsave("$data_vec_cosine_dir/cos_ken_cleaned_$(n_pairs_str)_noself.jld2"; cosine=clean_cosine, kendall=clean_kendall)

begin
    fig = Figure(size=(600, 500))
    ax = Axis(
        fig[1, 1],
        xlabel="Kendall Tau distance",
        ylabel="Euclidean distance")
    rx = (maximum(clean_kendall) - minimum(clean_kendall)) / 100
    ry = (maximum(clean_euclid) - minimum(clean_euclid)) / 100
    hb = hexbin!(ax, Float64.(clean_kendall), Float64.(clean_euclid), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="count (log10)")
    display(fig)
end
save("$fig_vec_euclid_dir/euc_ken_cleaned_$(n_pairs_str)_noself.png", fig)

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], xlabel="Kendall Tau distance", ylabel="Cosine distance")
            #   title="Pairwise distances in expression vs. ranked vectors")
    rx = (maximum(clean_kendall) - minimum(clean_kendall)) / 100
    ry = (maximum(clean_cosine) - minimum(clean_cosine)) / 100
    hb = hexbin!(ax, Float64.(clean_kendall), Float64.(clean_cosine), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="Count (log10)")
    display(fig)
end
# save("$fig_vec_cosine_dir/cosine_kendall_cleaned_100M.png", fig)
# jldsave("$data_vec_cosine_dir/cos_ken_cleaned_$(n_pairs_str)_noself.jld2"; cosine=clean_cosine, kendall=clean_kendall)

###################################################################################################################################

# confounder summary: overlaid histograms for each distance metric (3 figures)
# uses cleaned data (bad plate removed)

clean_plate_a = det_plates[idx_a[clean_mask]]
clean_plate_b = det_plates[idx_b[clean_mask]]
clean_pert_a  = pert_id[idx_a[clean_mask]]
clean_pert_b  = pert_id[idx_b[clean_mask]]

same_plate = clean_plate_a .== clean_plate_b
same_pert  = clean_pert_a .== clean_pert_b

# euclidean version
begin
    fig_conf_e = Figure(size=(700, 600))

    ax_ce1 = Axis(fig_conf_e[1, 1], xlabel="euclidean distance", ylabel="density", title="plate")
    hist!(ax_ce1, Float64.(clean_euclid[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_ce1, Float64.(clean_euclid[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_ce1, position=:lt)

    ax_ce2 = Axis(fig_conf_e[2, 1], xlabel="euclidean distance", ylabel="density", title="compound")
    hist!(ax_ce2, Float64.(clean_euclid[same_pert]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same compound (n=$(sum(same_pert)))")
    hist!(ax_ce2, Float64.(clean_euclid[.!same_pert]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff compound (n=$(sum(.!same_pert)))")
    axislegend(ax_ce2, position=:lt)

    display(fig_conf_e)
end
save("$fig_var_dir/confounder_euclidean_$(n_pairs_str).png", fig_conf_e)

# cosine version
begin
    fig_conf_c = Figure(size=(700, 600))

    ax_cc1 = Axis(fig_conf_c[1, 1], xlabel="cosine distance", ylabel="density", title="plate")
    hist!(ax_cc1, Float64.(clean_cosine[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_cc1, Float64.(clean_cosine[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_cc1, position=:rt)

    ax_cc2 = Axis(fig_conf_c[2, 1], xlabel="cosine distance", ylabel="density", title="compound")
    hist!(ax_cc2, Float64.(clean_cosine[same_pert]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same compound (n=$(sum(same_pert)))")
    hist!(ax_cc2, Float64.(clean_cosine[.!same_pert]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff compound (n=$(sum(.!same_pert)))")
    axislegend(ax_cc2, position=:rt)

    display(fig_conf_c)
end
save("$fig_var_dir/confounder_cosine_$(n_pairs_str).png", fig_conf_c)

# kendall tau version
begin
    fig_conf_k = Figure(size=(700, 600))

    ax_ck1 = Axis(fig_conf_k[1, 1], xlabel="kendall tau distance", ylabel="density", title="plate")
    hist!(ax_ck1, Float64.(clean_kendall[same_plate]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="same plate (n=$(sum(same_plate)))")
    hist!(ax_ck1, Float64.(clean_kendall[.!same_plate]), bins=100, normalization=:pdf, color=(:darkorange, 0.4), label="diff plate (n=$(sum(.!same_plate)))")
    axislegend(ax_ck1, position=:lt)

    ax_ck2 = Axis(fig_conf_k[2, 1], xlabel="kendall tau distance", ylabel="density", title="compound")
    hist!(ax_ck2, Float64.(clean_kendall[same_pert]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="same compound (n=$(sum(same_pert)))")
    hist!(ax_ck2, Float64.(clean_kendall[.!same_pert]), bins=100, normalization=:pdf, color=(:gray50, 0.4), label="diff compound (n=$(sum(.!same_pert)))")
    axislegend(ax_ck2, position=:lt)

    display(fig_conf_k)
end
save("$fig_var_dir/confounder_kendall_$(n_pairs_str).png", fig_conf_k)

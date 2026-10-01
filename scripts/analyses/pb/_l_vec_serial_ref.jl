using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../../..", arch_dir))
using JLD2, Statistics, StatsBase, CairoMakie, LinearAlgebra, Random, DataFrames

data = "data_expr.jld2"
expr = load("data/lincs/$data")["data_expr"]
# fig_vec_cosine_dir = "results/lincs/figures/vectors/cosine"
# fig_vec_euclid_dir = "results/lincs/figures/vectors/euclid"
# fig_var_dir = "results/lincs/figures/variables"
# data_vec_cosine_dir = "results/lincs/data/vectors/cosine"
# data_vec_euclid_dir = "results/lincs/data/vectors/euclid"
# knobs (defaults = 100K pairs); everything that depends on the sampled pairs goes into a <pairs> subfolder (100K, 1M, ...)
# VEC_NPAIRS=1000000 julia -t 32 scripts/analyses/pb/l_vec.jl
n_pairs = parse(Int, get(ENV, "VEC_NPAIRS", "100000"))
n_pairs_str = n_pairs >= 1_000_000 ? "$(div(n_pairs, 1_000_000))M" : "$(div(n_pairs, 1_000))K"
res_root = get(ENV, "VEC_ROOT", "results/lincs")
Random.seed!(parse(Int, get(ENV, "VEC_SEED", "42")))
fig_vec_cosine_dir = "$res_root/figures/vectors/cosine/$n_pairs_str"
fig_vec_euclid_dir = "$res_root/figures/vectors/euclid/$n_pairs_str"
fig_var_dir = "$res_root/figures/variables/$n_pairs_str"
data_vec_cosine_dir = "$res_root/data/vectors/cosine/$n_pairs_str"
data_vec_euclid_dir = "$res_root/data/vectors/euclid/$n_pairs_str"
save_prefix = "lincs"

n_genes, N = size(expr)
# train nonzero medians, detected first
gene_medians = let p = "data/lincs/gene_medians.jld2"
    if isfile(p) && length(load(p, "medians")) == size(expr, 1)
        Float32.(load(p, "medians"))
    else
        @warn "no usable medians file $p: computing nonzero medians here"
        Float32[(v = filter(>(0), r); isempty(v) ? 1f0 : median(v)) for r in eachrow(expr)]
    end
end
println("dataset=lincs  n_genes=$n_genes  N=$N")

mkpath(fig_vec_cosine_dir); mkpath(fig_vec_euclid_dir); mkpath(fig_var_dir)
mkpath(data_vec_cosine_dir); mkpath(data_vec_euclid_dir)


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

# spearman distance (1 - rho) / 2 on x / median (tied ranks averaged)
rank_spearman = Vector{Float32}(undef, n_pairs)
for k in 1:n_pairs
    rank_spearman[k] = (1f0 - Float32(corspearman(Float64.(view(expr, :, idx_a[k]) ./ gene_medians),
                                                  Float64.(view(expr, :, idx_b[k]) ./ gene_medians)))) / 2f0
end

# n_pairs_str = n_pairs >= 1_000_000 ? "$(div(n_pairs, 1_000_000))M" : "$(div(n_pairs, 1_000))K"


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
save("$fig_vec_euclid_dir/euclid_kendall_$(n_pairs_str)_noself.png", fig)

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
save("$fig_vec_cosine_dir/cosine_kendall_$(n_pairs_str)_noself.png", fig)


# LINCS upper blob diagnosis

mfc = load("data/lincs/lincs_trt_inst.jld2")["mfc"]
pert_id = load("data/lincs/lincs_trt_inst.jld2")["pert_id"]

inst = load("data/lincs/lincs_trt_data.jld2")["filtered_data"].inst
sample_ids = inst.sample_id
det_plates = inst.det_plate
cmap_names = inst.cmap_name

# upper tail frequency
tail_threshold = 0.10f0
tail_mask = expr_cosine .> tail_threshold
tail_indices = findall(tail_mask)

function tail_freq(ids, idx_a, idx_b, tail_indices)
    countmap(vcat(
        [ids[idx_a[k]] for k in tail_indices],
        [ids[idx_b[k]] for k in tail_indices]))
end

# compounds
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

# samples
tail_samples = tail_freq(sample_ids, idx_a, idx_b, tail_indices)
tail_samples_sorted = sort(collect(tail_samples), by=x->-x[2])
for (sid, count) in tail_samples_sorted[1:min(20, length(tail_samples_sorted))]
    println("$sid: $count")
end

# plates
tail_plates = tail_freq(det_plates, idx_a, idx_b, tail_indices)
tail_plates_sorted = sort(collect(tail_plates), by=x->-x[2])
overall_plates = countmap(det_plates)
for (plate, count) in tail_plates_sorted[1:min(20, length(tail_plates_sorted))]
    overall_count = get(overall_plates, plate, 0)
    overall_frac = round(100 * overall_count / length(det_plates), digits=2)
    println("$plate: $count tail appearances (overall: $overall_count samples, $overall_frac%)")
end

# upper blob is plate REP.A010_JURKAT_24H_X3_B32, removed below

bad_plate = Symbol("REP.A010_JURKAT_24H_X3_B32")
clean_mask = [(det_plates[idx_a[k]] != bad_plate) && (det_plates[idx_b[k]] != bad_plate) for k in 1:n_pairs]
println("pairs before: $n_pairs, after: $(sum(clean_mask))")

clean_cosine = expr_cosine[clean_mask]
clean_kendall = rank_kendall[clean_mask]
clean_euclid = expr_euclid[clean_mask]
clean_spearman = rank_spearman[clean_mask]

# jldsave("$data_vec_euclid_dir/euc_ken_cleaned_$(n_pairs_str)_noself.jld2"; euclid=clean_euclid, kendall=clean_kendall)
# jldsave("$data_vec_cosine_dir/cos_ken_cleaned_$(n_pairs_str)_noself.jld2"; cosine=clean_cosine, kendall=clean_kendall)
jldsave("$data_vec_euclid_dir/euc_ken_cleaned_$(n_pairs_str)_noself.jld2"; euclid=clean_euclid, kendall=clean_kendall, spearman=clean_spearman)
jldsave("$data_vec_cosine_dir/cos_ken_cleaned_$(n_pairs_str)_noself.jld2"; cosine=clean_cosine, kendall=clean_kendall, spearman=clean_spearman)

# spearman versions of the kendall hexbins (raw and cleaned)
for (sfx, cosv, eucv, spev) in (("", expr_cosine, expr_euclid, rank_spearman), ("_cleaned", clean_cosine, clean_euclid, clean_spearman))
    for (ename, ev, d) in (("cosine", cosv, fig_vec_cosine_dir), ("euclid", eucv, fig_vec_euclid_dir))
        local fig = Figure(size=(600, 450))
        local ax = Axis(fig[1, 1], xlabel="Spearman distance", ylabel=(ename == "cosine" ? "Cosine distance" : "Euclidean distance"))
        local rx = (maximum(spev) - minimum(spev) + 1f-6) / 100; local ry = (maximum(ev) - minimum(ev) + 1f-6) / 100
        local hb = hexbin!(ax, Float64.(spev), Float64.(ev), cellsize=(rx, ry), colorscale=log10)
        Colorbar(fig[1, 2], hb, label="Count (log10)")
        save("$d/$(ename)_spearman$(sfx)_$(n_pairs_str)_noself.png", fig)
    end
end
println("spearman(cosine, kendall) = $(round(corspearman(Float64.(clean_cosine), Float64.(clean_kendall)), digits=3)), spearman(cosine, spearman) = $(round(corspearman(Float64.(clean_cosine), Float64.(clean_spearman)), digits=3)) (cleaned)")

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
    # display(fig)
end
save("$fig_vec_euclid_dir/euc_ken_cleaned_$(n_pairs_str)_noself.png", fig)

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], xlabel="Kendall Tau distance", ylabel="Cosine distance")
    rx = (maximum(clean_kendall) - minimum(clean_kendall)) / 100
    ry = (maximum(clean_cosine) - minimum(clean_cosine)) / 100
    hb = hexbin!(ax, Float64.(clean_kendall), Float64.(clean_cosine), cellsize=(rx, ry), colorscale=log10)
    Colorbar(fig[1, 2], hb, label="Count (log10)")
    # display(fig)
end
save("$fig_vec_cosine_dir/cosine_kendall_cleaned_$(n_pairs_str)_noself.png", fig)


# confounder histograms (cleaned data)

clean_plate_a = det_plates[idx_a[clean_mask]]
clean_plate_b = det_plates[idx_b[clean_mask]]
clean_pert_a  = pert_id[idx_a[clean_mask]]
clean_pert_b  = pert_id[idx_b[clean_mask]]

same_plate = clean_plate_a .== clean_plate_b
same_pert  = clean_pert_a .== clean_pert_b

# euclidean
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

    # display(fig_conf_e)
end
save("$fig_var_dir/confounder_euclidean_$(n_pairs_str).png", fig_conf_e)

# cosine
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

    # display(fig_conf_c)
end
save("$fig_var_dir/confounder_cosine_$(n_pairs_str).png", fig_conf_c)

# kendall
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

    # display(fig_conf_k)
end
save("$fig_var_dir/confounder_kendall_$(n_pairs_str).png", fig_conf_k)

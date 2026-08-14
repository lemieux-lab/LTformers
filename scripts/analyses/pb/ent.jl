using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))
using JLD2, StatsBase, Statistics, CairoMakie, DataFrames

# dataset = "lincs"
dataset = "tahoe" # for pseudobulk

if dataset == "lincs"
    expr = load("data/data_expr.jld2")["data_expr"]
    fig_dir = "results/lincs/figures/entropies"
    data_dir = "results/lincs/data/entropies"
    save_prefix = "lincs"
elseif dataset == "tahoe"
    df = load("/home/muninn/scratch/kaufmanl/CAP/results/tahoe/pseudobulks/filtered_pseudobulks_alpha_10000.jld2")["df"]
    expr = hcat(df.expr...)
    fig_dir = "results/tahoe/pb/figures/entropies"
    data_dir = "results/tahoe/pb/data/entropies"
    save_prefix = "pb"
end

n_genes, N = size(expr)
gene_medians = vec(median(expr, dims=2)) .+ 1f-10

mkpath(fig_dir); mkpath(data_dir)

#######################################################################################################################################

function rank_genes(expr, medians)
    n, m = size(expr)
    data_ranked = Matrix{Int32}(undef, size(expr))
    normalized_col = Vector{Float32}(undef, n)
    sorted_ind_col = Vector{Int32}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        sortperm!(sorted_ind_col, normalized_col, rev=true)
        data_ranked[:, j] .= sorted_ind_col
    end
    return data_ranked
end

ranked = rank_genes(expr, gene_medians)

#######################################################################################################################################

### calculating entropy per rank position

function calculate_entropy(row)
    n = length(row)
    if n == 0
        return 0.0
    end
    counts_dict = countmap(row)
    probabilities = values(counts_dict) ./ n
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy
end

entropies = Float64[]
for row in eachrow(ranked)
    e = calculate_entropy(row)
    push!(entropies, e)
end

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        xtickformat=values -> [string(Int(round(v))) for v in values])
        # title="Shannon entropy by gene rank position")
    scatter!(ax, 1:n_genes, entropies, alpha=0.5, color=:black)
    display(fig)
end

save("$fig_dir/$(save_prefix)_rank_entropy.png", fig)
jldsave("$data_dir/$(save_prefix)_ranked_entropies.jld2"; entropies=entropies)

### sparsity per rank position (fraction of samples where gene at that rank has zero expression)

sparsities = Float64[]
for r in 1:n_genes
    n_zero = 0
    for j in 1:N
        if expr[ranked[r, j], j] == 0.0f0
            n_zero += 1
        end
    end
    push!(sparsities, n_zero / N)
end

jldsave("$data_dir/$(save_prefix)_ranked_sparsities.jld2"; sparsities=sparsities)

### entropy + sparsity overlay

begin
    fig_overlay = Figure(size=(600, 500))
    ax_ent = Axis(fig_overlay[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy (bits)",
        yaxisposition=:left,
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title = "Tahoe pseudo-bulk entropy vs. sparsity per rank position")
    ax_spar = Axis(fig_overlay[1, 1],
        ylabel="Sparsity (1 = always 0)",
        yaxisposition=:right)
    hidespines!(ax_spar)
    hidexdecorations!(ax_spar)

    scatter!(ax_ent, 1:n_genes, entropies, alpha=0.5, color=:black, markersize=4, label="Entropy")
    scatter!(ax_spar, 1:n_genes, sparsities, alpha=0.5, color=Makie.wong_colors()[1], markersize=4, label="Sparsity")

    # Legend(fig_overlay[0, 1],
    #     [MarkerElement(color=:black, marker=:circle, markersize=8),
    #      MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=8)],
    #     ["Entropy", "Sparsity"],
    #     orientation=:horizontal, tellwidth=false)

    display(fig_overlay)
end
save("$fig_dir/$(save_prefix)_rank_entropy_sparsity.png", fig_overlay)

### mean expression per gene (sorted by mean expression)

gene_means = vec(mean(expr, dims=2))
gene_std_devs = vec(std(expr, dims=2))
sorted_indices_by_mean = sortperm(gene_means, rev=true)
gene_indices = 1:n_genes

begin
    fig_mean = Figure(size=(600, 400))
    ax_mean = Axis(fig_mean[1, 1],
        xlabel="gene index (sorted by mean expression)",
        ylabel="mean expression level",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_mean, gene_indices, gene_means[sorted_indices_by_mean], alpha=0.5, markersize=5, color=Makie.wong_colors()[2])
    display(fig_mean)
end
save("$fig_dir/gene_exp_mean.png", fig_mean)

### std dev per gene (sorted by mean expression)

begin
    fig_std = Figure(size=(600, 400))
    ax_std = Axis(fig_std[1, 1],
        xlabel="gene index (sorted by mean expression)",
        ylabel="standard deviation",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_std, gene_indices, gene_std_devs[sorted_indices_by_mean], alpha=0.5, color=Makie.wong_colors()[3])
    display(fig_std)
end
save("$fig_dir/gene_exp_stddev.png", fig_std, px_per_unit=2)

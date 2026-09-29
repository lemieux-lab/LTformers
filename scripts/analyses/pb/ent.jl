using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))
using JLD2, StatsBase, Statistics, CairoMakie, DataFrames

# dataset = "tahoe"
dataset = isempty(ARGS) ? "tahoe" : ARGS[1]   # julia ent.jl [tahoe|lincs]
dataset in ("tahoe", "lincs") || error("dataset must be tahoe or lincs, got $dataset")

if dataset == "lincs"
    expr = load("data/lincs/data_expr.jld2")["data_expr"]
    fig_dir = "results/lincs/figures/entropies"
    data_dir = "results/lincs/data/entropies"
    save_prefix = "lincs"
elseif dataset == "tahoe"
    df = load("data/tahoe/filtered_pseudobulks_alpha_10000.jld2")["df"]
    expr = hcat(df.expr...)
    fig_dir = "results/tahoe/pb/figures/entropies"
    data_dir = "results/tahoe/pb/data/entropies"
    save_prefix = "pb"
end

n_genes, N = size(expr)
# train nonzero medians, detected first
gene_medians = let p = dataset == "lincs" ? "data/lincs/gene_medians.jld2" : "data/tahoe/pb_gene_medians.jld2"
    if isfile(p) && length(load(p, "medians")) == size(expr, 1)
        Float32.(load(p, "medians"))
    else
        @warn "no usable medians file $p: computing nonzero medians here"
        Float32[(v = filter(>(0), r); isempty(v) ? 1f0 : median(v)) for r in eachrow(expr)]
    end
end

mkpath(fig_dir); mkpath(data_dir)


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


# entropy per rank

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

# detected genes only
n_det = vec(sum(expr .> 0, dims=1))
# entropies = Float64[]
# for r in 1:n_genes
#     push!(entropies, calculate_entropy(ranked[r, n_det .>= r]))
# end
# ranks no sample reaches (no detected gene there) are undefined -> NaN, not 0 (calculate_entropy returns 0 for empty)
n_at_rank = [count(>=(r), n_det) for r in 1:n_genes]   # samples with a detected gene at rank r
entropies = [n_at_rank[r] == 0 ? NaN : calculate_entropy(ranked[r, n_det .>= r]) for r in 1:n_genes]
# normalized entropy: H / max possible H at that rank, log2(min(samples counted, genes)); removes the drop that comes
# only from fewer samples reaching deep ranks. undefined for < 2 samples
norm_entropies = [n_at_rank[r] < 2 ? NaN : entropies[r] / log2(min(n_at_rank[r], n_genes)) for r in 1:n_genes]
last_defined = findlast(>(0), n_at_rank)
println("entropy defined up to rank $last_defined (max detected genes per sample); ranks beyond are NaN")

begin
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax, 1:n_genes, entropies, alpha=0.5, color=:black)
    display(fig)
end

save("$fig_dir/$(save_prefix)_rank_entropy.png", fig)
# jldsave("$data_dir/$(save_prefix)_ranked_entropies.jld2"; entropies=entropies)
jldsave("$data_dir/$(save_prefix)_ranked_entropies.jld2"; entropies=entropies, norm_entropies=norm_entropies, n_at_rank=n_at_rank)

# normalized entropy per rank (separate plot)
begin
    fig_norm = Figure(size=(600, 500))
    ax_norm = Axis(fig_norm[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Normalized Shannon entropy (H / log2(n samples))",
        xtickformat=values -> [string(Int(round(v))) for v in values])
    scatter!(ax_norm, 1:n_genes, norm_entropies, alpha=0.5, color=:black)
    ylims!(ax_norm, 0, 1.05)
    display(fig_norm)
end
save("$fig_dir/$(save_prefix)_rank_entropy_normalized.png", fig_norm)

# sparsity per rank

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

# unique count diversity per rank

unique_diversity_sum = zeros(Float64, n_genes)
for j in 1:N
    seen = Set{Float32}()
    for r in n_genes:-1:1
        push!(seen, expr[ranked[r, j], j])
        unique_diversity_sum[r] += length(seen)
    end
end
unique_diversity = unique_diversity_sum ./ N
unique_diversity_norm = unique_diversity ./ unique_diversity[1]

jldsave("$data_dir/$(save_prefix)_ranked_unique_diversity.jld2"; unique_diversity=unique_diversity, unique_diversity_norm=unique_diversity_norm)

# entropy + sparsity overlay

begin
    fig_overlay = Figure(size=(600, 500))
    ax_ent = Axis(fig_overlay[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy (bits)",
        yaxisposition=:left,
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title = dataset == "lincs" ? "LINCS L1000 entropy vs. sparsity per rank position" : "Tahoe pseudo-bulk entropy vs. sparsity per rank position")
    ax_spar = Axis(fig_overlay[1, 1],
        ylabel="Sparsity (1 = always 0)",
        yaxisposition=:right)
    hidespines!(ax_spar)
    hidexdecorations!(ax_spar)

    scatter!(ax_ent, 1:n_genes, entropies, alpha=0.5, color=:black, markersize=4, label="Entropy")
    scatter!(ax_spar, 1:n_genes, sparsities, alpha=0.5, color=Makie.wong_colors()[1], markersize=4, label="Sparsity")


    display(fig_overlay)
end
save("$fig_dir/$(save_prefix)_rank_entropy_sparsity.png", fig_overlay)

# unique count diversity (normalized)

begin
    fig_ud = Figure(size=(600, 500))
    ax_ud = Axis(fig_ud[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Normalized unique count diversity",
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title = dataset == "lincs" ? "LINCS L1000 unique count diversity per rank" : "Tahoe pseudo-bulk unique count diversity per rank")
    lines!(ax_ud, 1:n_genes, unique_diversity_norm, linewidth=2, color=:black)
    display(fig_ud)
end
save("$fig_dir/$(save_prefix)_rank_unique_diversity.png", fig_ud)

# mean expression per gene

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

# std per gene

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

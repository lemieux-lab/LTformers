module Plot

using CairoMakie, StatsBase, Statistics, JLD2

export plot_loss, plot_ranked_heatmap, plot_cosine
export plot_per_rank_error, plot_per_gene_error, plot_per_sample_rank_error


function plot_loss(n_epochs::Int, train_losses, test_losses, save_dir::String, loss::String;
                   val_losses=Float32[])
    fig_loss = Figure(size = (600, 400))
    ax_loss = Axis(fig_loss[1, 1], xlabel="Epoch", ylabel="Loss ($loss)")
    lines!(ax_loss, 1:n_epochs, train_losses, label="Train", linewidth=2)
    if !isempty(val_losses) && length(val_losses) == n_epochs
        lines!(ax_loss, 1:n_epochs, val_losses, label="Val", linewidth=2)
    end
    if length(test_losses) == n_epochs
        lines!(ax_loss, 1:n_epochs, test_losses, label="Test", linewidth=2)
    elseif length(test_losses) >= 1
        # test evaluated only at end — plot as marker(s) at the correct epoch(s)
        test_epochs = (n_epochs - length(test_losses) + 1):n_epochs
        scatter!(ax_loss, collect(test_epochs), collect(test_losses),
                 label="Test", markersize=8)
    end
    axislegend(ax_loss, position=:rt)
    save("$save_dir/loss.png", fig_loss)
end


function plot_ranked_heatmap(all_trues, all_preds, save_dir::String)
    cs = corspearman(all_trues, all_preds)
    cp = cor(all_trues, all_preds)

    n_genes = max(maximum(all_trues), maximum(all_preds))
    bin_edges = 1:(n_genes + 1)
    h = fit(Histogram, (all_trues, all_preds), (bin_edges, bin_edges))

    fig_hm = Figure(size = (600, 400))
    ax_hm = Axis(fig_hm[1, 1], xlabel="true rank", ylabel="predicted rank")

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align=(:left, :top), text="Pearson: $cp", color=:white)
    Colorbar(fig_hm[1, 2], hm, label="count (log10)")

    save(joinpath(save_dir, "heatmap.png"), fig_hm)
    return cs, cp
end

function plot_cosine(n_epochs::Int, test_cos_sims, save_dir::String)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="epoch", ylabel="cosine similarity")
    lines!(ax, 1:n_epochs, test_cos_sims, linewidth=2)
    save("$save_dir/cos_sim_plot.png", fig)
end


function plot_per_rank_error(rank_error_sums, rank_error_counts, n_genes, save_dir)
    mean_errors = Float32[]
    active_ranks = Int[]
    for r in 1:n_genes
        if rank_error_counts[r] > 0
            push!(mean_errors, rank_error_sums[r] / rank_error_counts[r])
            push!(active_ranks, r)
        end
    end

    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="true rank", ylabel="mean rank error (# logits above true)")
    lines!(ax, active_ranks, mean_errors, linewidth=1, color=:steelblue)
    overall = sum(rank_error_sums) / max(sum(rank_error_counts), 1)
    text!(ax, 0.95, 0.95, space=:relative, align=(:right, :top),
        text="overall mean: $(round(overall, digits=2))")
    save(joinpath(save_dir, "per_rank_error.png"), fig)

    jldsave(joinpath(save_dir, "per_rank_error.jld2");
        rank_error_sums=rank_error_sums, rank_error_counts=rank_error_counts)
end

function plot_per_gene_error(error_sums, error_counts, n_genes, save_dir, ylabel, filename;
                             sorted_gene_path::String="")
    mean_errors = zeros(Float32, n_genes)
    for i in 1:n_genes
        if error_counts[i] > 0
            mean_errors[i] = error_sums[i] / error_counts[i]
        end
    end

    if !isempty(sorted_gene_path) && isfile(sorted_gene_path)
        sorted_indices_by_mean = load(sorted_gene_path)["sorted_indices_by_mean"]
        gene_to_rank = invperm(sorted_indices_by_mean)

        ranked_errors = zeros(Float32, n_genes)
        for i in 1:n_genes
            ranked_errors[gene_to_rank[i]] = mean_errors[i]
        end

        fig = Figure(size=(800, 400))
        ax = Axis(fig[1, 1], xlabel="expression rank (global median)", ylabel=ylabel)
        active = findall(ranked_errors .> 0)
        lines!(ax, active, ranked_errors[active], linewidth=1, color=:steelblue)
        overall = sum(error_sums) / max(sum(error_counts), 1)
        text!(ax, 0.95, 0.95, space=:relative, align=(:right, :top),
            text="overall mean: $(round(overall, digits=4))")
        save(joinpath(save_dir, filename * "_global.png"), fig)
    end

    fig2 = Figure(size=(800, 400))
    ax2 = Axis(fig2[1, 1], xlabel="gene position", ylabel=ylabel)
    active2 = findall(mean_errors .> 0)
    overall2 = sum(error_sums) / max(sum(error_counts), 1)
    lines!(ax2, active2, mean_errors[active2], linewidth=1, color=:steelblue)
    text!(ax2, 0.95, 0.95, space=:relative, align=(:right, :top),
        text="overall mean: $(round(overall2, digits=4))")
    save(joinpath(save_dir, filename * "_position.png"), fig2)

    jldsave(joinpath(save_dir, filename * ".jld2");
        error_sums=error_sums, error_counts=error_counts)
end

function plot_per_sample_rank_error(error_sums, error_counts, n_genes, save_dir, ylabel, filename)
    mean_errors = Float32[]
    active_ranks = Int[]
    for r in 1:n_genes
        if error_counts[r] > 0
            push!(mean_errors, error_sums[r] / error_counts[r])
            push!(active_ranks, r)
        end
    end

    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="within-sample rank", ylabel=ylabel)
    lines!(ax, active_ranks, mean_errors, linewidth=1, color=:steelblue)
    overall = sum(error_sums) / max(sum(error_counts), 1)
    text!(ax, 0.95, 0.95, space=:relative, align=(:right, :top),
        text="overall mean: $(round(overall, digits=4))")
    save(joinpath(save_dir, filename * "_sample.png"), fig)

    jldsave(joinpath(save_dir, filename * "_sample.jld2");
        error_sums=error_sums, error_counts=error_counts)
end


end  # module Plot

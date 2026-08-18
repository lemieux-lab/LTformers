using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../..", arch_dir)))
using JLD2, CairoMakie, Statistics

lincs_data = "results/lincs/data/vectors"
tahoe_data = "results/tahoe/pb/data/vectors"
sc_data = "results/tahoe/sc/data/vectors"

# cosine & kendall
lincs = load("$lincs_data/cosine/cos_ken_cleaned_100K_noself.jld2")
tahoe = load("$tahoe_data/cos_ken_100K_noself.jld2")
sc = load("$sc_data/sc_distances_100000.jld2")

# euclidean & kendall
lincs_euc = load("$lincs_data/euclid/euc_ken_cleaned_100K_noself.jld2")
tahoe_euc = load("$tahoe_data/euc_ken_100K_noself.jld2")
# sc uses same file for all distance types
sc_euc = sc

fig_dir = "results/tahoe/pb/figures/lincs_comparison"
fig_cos_dir = "$fig_dir/cosine"
fig_euc_dir = "$fig_dir/euclid"
fig_ent_dir = "$fig_dir/entropy"
mkpath(fig_cos_dir); mkpath(fig_euc_dir); mkpath(fig_ent_dir)

#######################################################################################################################################

# stacked hexbin: cosine vs kendall

begin
    fig = Figure(size=(700, 1200))

    # LINCS (top)
    ax1 = Axis(fig[1, 1],
        ylabel="cosine distance",
        title="LINCS (978 genes, 100K pairs)")
    hidexdecorations!(ax1, grid=false)
    rx1 = (maximum(lincs["kendall"]) - minimum(lincs["kendall"])) / 100
    ry1 = (maximum(lincs["cosine"]) - minimum(lincs["cosine"])) / 100
    hb1 = hexbin!(ax1, Float64.(lincs["kendall"]), Float64.(lincs["cosine"]),
        cellsize=(rx1, ry1), colorscale=log10)
    Colorbar(fig[1, 2], hb1, label="count (log10)")

    # Tahoe PB (middle)
    ax2 = Axis(fig[2, 1],
        ylabel="cosine distance",
        title="Tahoe pseudo-bulk (19,020 genes, 100K pairs)")
    hidexdecorations!(ax2, grid=false)
    rx2 = (maximum(tahoe["kendall"]) - minimum(tahoe["kendall"])) / 100
    ry2 = (maximum(tahoe["cosine"]) - minimum(tahoe["cosine"])) / 100
    hb2 = hexbin!(ax2, Float64.(tahoe["kendall"]), Float64.(tahoe["cosine"]),
        cellsize=(rx2, ry2), colorscale=log10)
    Colorbar(fig[2, 2], hb2, label="count (log10)")

    # Tahoe SC (bottom)
    ax3 = Axis(fig[3, 1],
        xlabel="kendall tau distance",
        ylabel="cosine distance",
        title="Tahoe single-cell (19,020 genes, 100K pairs)")
    rx3 = (maximum(sc["kendall"]) - minimum(sc["kendall"])) / 100
    ry3 = (maximum(sc["cosine"]) - minimum(sc["cosine"])) / 100
    hb3 = hexbin!(ax3, Float64.(sc["kendall"]), Float64.(sc["cosine"]),
        cellsize=(rx3, ry3), colorscale=log10)
    Colorbar(fig[3, 2], hb3, label="count (log10)")

    # link x-axes
    x_lo = min(minimum(lincs["kendall"]), minimum(tahoe["kendall"]), minimum(sc["kendall"]))
    x_hi = max(maximum(lincs["kendall"]), maximum(tahoe["kendall"]), maximum(sc["kendall"]))
    xlims!(ax1, x_lo, x_hi)
    xlims!(ax2, x_lo, x_hi)
    xlims!(ax3, x_lo, x_hi)

    display(fig)
end
save("$fig_cos_dir/lts_cosine_kendall_100K.png", fig)

#######################################################################################################################################

# stacked hexbin: euclid vs kendall

begin
    fig = Figure(size=(700, 1200))

    # LINCS (top)
    ax1 = Axis(fig[1, 1],
        ylabel="euclidean distance",
        title="LINCS L1000 (978 genes, 100K pairs)")
    hidexdecorations!(ax1, grid=false)
    rx1 = (maximum(lincs_euc["kendall"]) - minimum(lincs_euc["kendall"])) / 100
    ry1 = (maximum(lincs_euc["euclid"]) - minimum(lincs_euc["euclid"])) / 100
    hb1 = hexbin!(ax1, Float64.(lincs_euc["kendall"]), Float64.(lincs_euc["euclid"]),
        cellsize=(rx1, ry1), colorscale=log10)
    Colorbar(fig[1, 2], hb1, label="count (log10)")

    # Tahoe PB (middle)
    ax2 = Axis(fig[2, 1],
        ylabel="euclidean distance",
        title="Tahoe pseudobulk (19,020 genes, 100K pairs)")
    hidexdecorations!(ax2, grid=false)
    rx2 = (maximum(tahoe_euc["kendall"]) - minimum(tahoe_euc["kendall"])) / 100
    ry2 = (maximum(tahoe_euc["euclid"]) - minimum(tahoe_euc["euclid"])) / 100
    hb2 = hexbin!(ax2, Float64.(tahoe_euc["kendall"]), Float64.(tahoe_euc["euclid"]),
        cellsize=(rx2, ry2), colorscale=log10)
    Colorbar(fig[2, 2], hb2, label="count (log10)")

    # Tahoe SC (bottom)
    ax3 = Axis(fig[3, 1],
        xlabel="kendall tau distance",
        ylabel="euclidean distance",
        title="Tahoe single-cell (19,020 genes, 100K pairs)")
    rx3 = (maximum(sc_euc["kendall"]) - minimum(sc_euc["kendall"])) / 100
    ry3 = (maximum(sc_euc["euclidean"]) - minimum(sc_euc["euclidean"])) / 100
    hb3 = hexbin!(ax3, Float64.(sc_euc["kendall"]), Float64.(sc_euc["euclidean"]),
        cellsize=(rx3, ry3), colorscale=log10)
    Colorbar(fig[3, 2], hb3, label="count (log10)")

    # link x-axes
    x_lo = min(minimum(lincs_euc["kendall"]), minimum(tahoe_euc["kendall"]), minimum(sc_euc["kendall"]))
    x_hi = max(maximum(lincs_euc["kendall"]), maximum(tahoe_euc["kendall"]), maximum(sc_euc["kendall"]))
    xlims!(ax1, x_lo, x_hi)
    xlims!(ax2, x_lo, x_hi)
    xlims!(ax3, x_lo, x_hi)

    display(fig)
end
save("$fig_euc_dir/lts_euclid_kendall_100K.png", fig)

#######################################################################################################################################

# stacked overlaid histograms for each metric

begin
    fig2 = Figure(size=(700, 750))

    ax_k = Axis(fig2[1, 1], xlabel="kendall tau distance", ylabel="density", title="kendall tau distance")
    hist!(ax_k, Float64.(lincs["kendall"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS (n=$(length(lincs["kendall"])))")
    hist!(ax_k, Float64.(tahoe["kendall"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB (n=$(length(tahoe["kendall"])))")
    hist!(ax_k, Float64.(sc["kendall"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC (n=$(length(sc["kendall"])))")
    axislegend(ax_k, position=:rt)

    ax_c = Axis(fig2[2, 1], xlabel="cosine distance", ylabel="density", title="cosine distance")
    hist!(ax_c, Float64.(lincs["cosine"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS")
    hist!(ax_c, Float64.(tahoe["cosine"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB")
    hist!(ax_c, Float64.(sc["cosine"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC")
    axislegend(ax_c, position=:rt)

    ax_e = Axis(fig2[3, 1], xlabel="euclidean distance", ylabel="density", title="euclidean distance")
    hist!(ax_e, Float64.(lincs_euc["euclid"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS")
    hist!(ax_e, Float64.(tahoe_euc["euclid"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB")
    hist!(ax_e, Float64.(sc_euc["euclidean"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC")
    axislegend(ax_e, position=:rt)

    save("$fig_dir/compare_histograms_100K.png", fig2)
    display(fig2)
end

#######################################################################################################################################

# entropy comparison

lincs_ent = load("results/lincs/data/entropies/ranked_lincs_entropies.jld2")
tahoe_ent = load("results/tahoe/pb/data/entropies/ranked_pseudobulks_a10k_entropies.jld2")

lincs_H = lincs_ent["entropies"]
tahoe_H = tahoe_ent["entropies"]

# entropy by rank position (overlaid)

begin
    fig_rank = Figure(size=(700, 400))

    ax = Axis(fig_rank[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy (bits)")

    scatter!(ax, 1:length(tahoe_H), Float64.(tahoe_H),
        markersize=3, alpha=0.5, color=:teal, label="Tahoe ($(length(tahoe_H)) genes)")
    scatter!(ax, 1:length(lincs_H), Float64.(lincs_H),
        markersize=3, alpha=0.5, color=:orange, label="LINCS ($(length(lincs_H)) genes)")
    axislegend(ax, position=:rb)

    save("$fig_ent_dir/lt_entropy_ranked.png", fig_rank)
    display(fig_rank)
end

#######################################################################################################################################

# entropy/sparsity comparison

tahoe_spar = load("results/tahoe/pb/data/entropies/pb_ranked_sparsities.jld2")
sc_ent = load("results/tahoe/sc/data/entropies/ranked_sc_entropies.jld2")
sc_spar = load("results/tahoe/sc/data/entropies/ranked_sc_sparsities.jld2")

pb_S = tahoe_spar["sparsities"]
sc_H = sc_ent["entropies"]
sc_S = sc_spar["sparsities"]

n_genes = length(tahoe_H)  # 19020, same for all

begin
    fig_overlay = Figure(size=(700, 500))

    ax_ent = Axis(fig_overlay[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy (bits)",
        yaxisposition=:left,
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title="Tahoe entropy & sparsity per rank position (5 parquets sampled for SC)")
    ax_spar = Axis(fig_overlay[1, 1],
        ylabel="Sparsity (1 = always 0)",
        yaxisposition=:right)
    hidespines!(ax_spar)
    hidexdecorations!(ax_spar)

    # entropy: solid lines
    scatter!(ax_ent, 1:n_genes, Float64.(tahoe_H),
        markersize=4, alpha=0.5, color=Makie.wong_colors()[1], label="PB entropy")
    scatter!(ax_ent, 1:n_genes, Float64.(sc_H),
        markersize=4, alpha=0.5, color=Makie.wong_colors()[2], label="SC entropy")

    # sparsity: dotted lines (using lines! with linestyle)
    lines!(ax_spar, 1:n_genes, Float64.(pb_S),
        linewidth=3, linestyle=:dash, color=Makie.wong_colors()[1], label="PB sparsity")
    lines!(ax_spar, 1:n_genes, Float64.(sc_S),
        linewidth=3, linestyle=:dash, color=Makie.wong_colors()[2], label="SC sparsity")

    Legend(fig_overlay[0, 1],
        [MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=8),
         MarkerElement(color=Makie.wong_colors()[2], marker=:circle, markersize=8),
         LineElement(color=Makie.wong_colors()[1], linestyle=:dash, linewidth=2),
         LineElement(color=Makie.wong_colors()[2], linestyle=:dash, linewidth=2)],
        ["PB entropy", "SC entropy", "PB sparsity", "SC sparsity"],
        orientation=:horizontal, tellwidth=false, tellheight=true)

        display(fig_overlay)
    end
    save("$fig_ent_dir/pb_vs_sc_entropy_sparsity.png", fig_overlay)
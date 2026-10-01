using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../..", arch_dir)))
using JLD2, CairoMakie, Statistics

# Tahoe pseudobulk (TPB) vs Tahoe single cell (TSC) only; LINCS plots stand on their own (l_vec.jl, ent.jl lincs)
# lincs_data = "results/lincs/data/vectors"
# tahoe_data = "results/tahoe/pb/data/vectors"
# sc_data = "results/tahoe/sc/data/vectors"
# pairs tag = subfolder written by t_vec.jl / entvec.jl (100K, 1M, ...): CMP_PAIRS=1M julia scripts/analyses/comparison.jl
pairs_tag = get(ENV, "CMP_PAIRS", "100K")
tahoe_data = "results/tahoe/pb/data/vectors/$pairs_tag"
sc_data = "results/tahoe/sc/data/vectors/$pairs_tag"
# sc file names carry the exact pair count after self-pairs are dropped (e.g. 99999): find the file by pattern
sc_file(suffix) = joinpath(sc_data, only(filter(f -> occursin(Regex("^sc_distances_\\d+$(suffix)\\.jld2\$"), f), readdir(sc_data))))

# cosine & kendall
# lincs = load("$lincs_data/cosine/cos_ken_cleaned_100K_noself.jld2")
# tahoe = load("$tahoe_data/cos_ken_100K_noself.jld2")
tahoe = load("$tahoe_data/cos_ken_$(pairs_tag)_noself.jld2")
# sc = load("$sc_data/sc_distances_100000.jld2")
# sc = load("$sc_data/300_pqs/sc_distances_99999.jld2")   # entvec.jl, 300 parquets (Sep 30)
sc = load(sc_file(""))   # entvec.jl, 300 parquets

# euclidean & kendall
# lincs_euc = load("$lincs_data/euclid/euc_ken_cleaned_100K_noself.jld2")
# tahoe_euc = load("$tahoe_data/euc_ken_100K_noself.jld2")
# t_vec.jl now saves euclidean in the same file as cosine / kendall
tahoe_euc = Dict("euclid" => tahoe["euclidean"], "kendall" => tahoe["kendall"])
# sc: same file for all metrics
sc_euc = sc

# fig_dir = "results/tahoe/pb/figures/lincs_comparison"
fig_dir = "results/tahoe/pb_vs_sc/figures"
# fig_cos_dir = "$fig_dir/cosine"
# fig_euc_dir = "$fig_dir/euclid"
fig_cos_dir = "$fig_dir/cosine/$pairs_tag"
fig_euc_dir = "$fig_dir/euclid/$pairs_tag"
fig_hist_dir = "$fig_dir/histograms/$pairs_tag"
mkpath(fig_hist_dir)
fig_ent_dir = "$fig_dir/entropy"
mkpath(fig_cos_dir); mkpath(fig_euc_dir); mkpath(fig_ent_dir)


# hexbin: cosine vs kendall

# begin
    # fig = Figure(size=(700, 800))

    # fig = Figure(size=(700, 1200))

    # LINCS
    # ax1 = Axis(fig[1, 1],
        # ylabel="cosine distance",
        # title="LINCS (978 genes, 100K pairs)")
    # hidexdecorations!(ax1, grid=false)
    # rx1 = (maximum(lincs["kendall"]) - minimum(lincs["kendall"])) / 100
    # ry1 = (maximum(lincs["cosine"]) - minimum(lincs["cosine"])) / 100
    # hb1 = hexbin!(ax1, Float64.(lincs["kendall"]), Float64.(lincs["cosine"]),
        # cellsize=(rx1, ry1), colorscale=log10)
    # Colorbar(fig[1, 2], hb1, label="count (log10)")

    # Tahoe PB
    # ax2 = Axis(fig[2, 1],
    # ax2 = Axis(fig[1, 1],
        # ylabel="cosine distance",
        # ylabel="Cosine distance",
        # title="Tahoe pseudo-bulk (19,020 genes, 100K pairs)")
        # title="Tahoe pseudobulk (19,020 genes, 100K pairs)")
    # hidexdecorations!(ax2, grid=false)
    # rx2 = (maximum(tahoe["kendall"]) - minimum(tahoe["kendall"])) / 100
    # ry2 = (maximum(tahoe["cosine"]) - minimum(tahoe["cosine"])) / 100
    # hb2 = hexbin!(ax2, Float64.(tahoe["kendall"]), Float64.(tahoe["cosine"]),
        # cellsize=(rx2, ry2), colorscale=log10)
    # Colorbar(fig[2, 2], hb2, label="count (log10)")
    # Colorbar(fig[1, 2], hb2, label="Count (log10)")

    # Tahoe SC
    # ax3 = Axis(fig[3, 1],
    # ax3 = Axis(fig[2, 1],
        # xlabel="kendall tau distance",
        # xlabel="Kendall tau distance",
        # ylabel="cosine distance",
        # ylabel="Cosine distance",
        # title="Tahoe single-cell (19,020 genes, 100K pairs)")
        # title="Tahoe single cell (19,020 genes, 100K pairs)")
    # rx3 = (maximum(sc["kendall"]) - minimum(sc["kendall"])) / 100
    # ry3 = (maximum(sc["cosine"]) - minimum(sc["cosine"])) / 100
    # hb3 = hexbin!(ax3, Float64.(sc["kendall"]), Float64.(sc["cosine"]),
        # cellsize=(rx3, ry3), colorscale=log10)
    # Colorbar(fig[3, 2], hb3, label="count (log10)")
    # Colorbar(fig[2, 2], hb3, label="Count (log10)")

    # link x-axes
    # x_lo = min(minimum(lincs["kendall"]), minimum(tahoe["kendall"]), minimum(sc["kendall"]))
    # x_hi = max(maximum(lincs["kendall"]), maximum(tahoe["kendall"]), maximum(sc["kendall"]))
    # xlims!(ax1, x_lo, x_hi)
    # x_lo = min(minimum(tahoe["kendall"]), minimum(sc["kendall"]))
    # x_hi = max(maximum(tahoe["kendall"]), maximum(sc["kendall"]))
    # xlims!(ax2, x_lo, x_hi)
    # xlims!(ax3, x_lo, x_hi)

    # display(fig)
# end
# save("$fig_cos_dir/lts_cosine_kendall_100K.png", fig)
# save("$fig_cos_dir/ts_cosine_kendall_100K.png", fig)


# hexbin: euclid vs kendall

# begin
    # fig = Figure(size=(700, 1200))

    # LINCS
    # ax1 = Axis(fig[1, 1],
        # ylabel="euclidean distance",
        # title="LINCS L1000 (978 genes, 100K pairs)")
    # hidexdecorations!(ax1, grid=false)
    # rx1 = (maximum(lincs_euc["kendall"]) - minimum(lincs_euc["kendall"])) / 100
    # ry1 = (maximum(lincs_euc["euclid"]) - minimum(lincs_euc["euclid"])) / 100
    # hb1 = hexbin!(ax1, Float64.(lincs_euc["kendall"]), Float64.(lincs_euc["euclid"]),
        # cellsize=(rx1, ry1), colorscale=log10)
    # Colorbar(fig[1, 2], hb1, label="count (log10)")
    # fig = Figure(size=(700, 800))


    # Tahoe PB
    # ax2 = Axis(fig[2, 1],
    # ax2 = Axis(fig[1, 1],
        # ylabel="euclidean distance",
        # ylabel="Euclidean distance",
        # title="Tahoe pseudobulk (19,020 genes, 100K pairs)")
    # hidexdecorations!(ax2, grid=false)
    # rx2 = (maximum(tahoe_euc["kendall"]) - minimum(tahoe_euc["kendall"])) / 100
    # ry2 = (maximum(tahoe_euc["euclid"]) - minimum(tahoe_euc["euclid"])) / 100
    # hb2 = hexbin!(ax2, Float64.(tahoe_euc["kendall"]), Float64.(tahoe_euc["euclid"]),
        # cellsize=(rx2, ry2), colorscale=log10)
    # Colorbar(fig[2, 2], hb2, label="count (log10)")
    # Colorbar(fig[1, 2], hb2, label="Count (log10)")

    # Tahoe SC
    # ax3 = Axis(fig[3, 1],
    # ax3 = Axis(fig[2, 1],
        # xlabel="kendall tau distance",
        # xlabel="Kendall tau distance",
        # ylabel="euclidean distance",
        # ylabel="Euclidean distance",
        # title="Tahoe single-cell (19,020 genes, 100K pairs)")
        # title="Tahoe single cell (19,020 genes, 100K pairs)")
    # rx3 = (maximum(sc_euc["kendall"]) - minimum(sc_euc["kendall"])) / 100
    # ry3 = (maximum(sc_euc["euclidean"]) - minimum(sc_euc["euclidean"])) / 100
    # hb3 = hexbin!(ax3, Float64.(sc_euc["kendall"]), Float64.(sc_euc["euclidean"]),
        # cellsize=(rx3, ry3), colorscale=log10)
    # Colorbar(fig[3, 2], hb3, label="count (log10)")
    # Colorbar(fig[2, 2], hb3, label="Count (log10)")

    # link x-axes
    # x_lo = min(minimum(lincs_euc["kendall"]), minimum(tahoe_euc["kendall"]), minimum(sc_euc["kendall"]))
    # x_hi = max(maximum(lincs_euc["kendall"]), maximum(tahoe_euc["kendall"]), maximum(sc_euc["kendall"]))
    # xlims!(ax1, x_lo, x_hi)
    # x_lo = min(minimum(tahoe_euc["kendall"]), minimum(sc_euc["kendall"]))
    # x_hi = max(maximum(tahoe_euc["kendall"]), maximum(sc_euc["kendall"]))
    # xlims!(ax2, x_lo, x_hi)
    # xlims!(ax3, x_lo, x_hi)

    # display(fig)
# end
# save("$fig_euc_dir/lts_euclid_kendall_100K.png", fig)
# save("$fig_euc_dir/ts_euclid_kendall_100K.png", fig)

# hexbin: expression distance vs kendall, pseudobulk (top) and single cell (bottom), one figure per gene set
# detected = count > 0 (pseudobulk: summed counts; single cell: UMI count). r = Pearson correlation of the two distances
function pb_sc_hexbin(pb_x, pb_y, sc_x, sc_y; xlabel, ylabel, subtitle, path)
    fig = Figure(size=(700, 860))
    Label(fig[0, 1:2], subtitle, fontsize=12, color=:gray30, tellwidth=false)
    axs = Axis[]
    for (row, (x, y, name)) in enumerate([(pb_x, pb_y, "Tahoe pseudobulk"), (sc_x, sc_y, "Tahoe single cell")])
        x = Float64.(x); y = Float64.(y)
        ax = Axis(fig[row, 1], xlabel=(row == 2 ? xlabel : ""), ylabel=ylabel,
            # title="$name ($(round(Int, length(x) / 1000))K pairs), r = $(round(cor(x, y), digits=2))")
            title="$name ($pairs_tag pairs), r = $(round(cor(x, y), digits=2))")
        row == 1 && hidexdecorations!(ax, grid=false)
        hb = hexbin!(ax, x, y, cellsize=((maximum(x) - minimum(x)) / 100, (maximum(y) - minimum(y)) / 100), colorscale=log10)
        Colorbar(fig[row, 2], hb, label="Count (log10)")
        push!(axs, ax)
    end
    linkxaxes!(axs...)
    save(path, fig)
end

# tahoe_top = load("$tahoe_data/pb_distances_top1024_100K_noself.jld2")
# tahoe_hvg = load("$tahoe_data/pb_distances_hvg1024_100K_noself.jld2")
# sc_top = load("$sc_data/300_pqs/sc_distances_99999_top1024.jld2")
# sc_hvg = load("$sc_data/300_pqs/sc_distances_99999_hvg1024.jld2")
tahoe_top = load("$tahoe_data/pb_distances_top1024_$(pairs_tag)_noself.jld2")
tahoe_hvg = load("$tahoe_data/pb_distances_hvg1024_$(pairs_tag)_noself.jld2")
sc_top = load(sc_file("_top1024"))
sc_hvg = load(sc_file("_hvg1024"))
# model-matched needs the same pairs in the top-1024 and HVG files
@assert tahoe_top["idx_a"] == tahoe_hvg["idx_a"] && sc_top["idx_a"] == sc_hvg["idx_a"]

# (tag, subtitle, pb expression file, pb rank file, sc expression file, sc rank file)
gene_sets = [
    ("all", "All 19,020 genes; undetected genes (count = 0) tied at the lowest rank (rlog / rmlp full input)",
        tahoe, tahoe, sc, sc),
    ("top1024", "Top 1,024 detected genes (count > 0) per sample; genes outside the list tied (RTF and rlog / rmlp input)",
        tahoe_top, tahoe_top, sc_top, sc_top),
    ("hvg1024", "1,024 highly variable genes; undetected genes (count = 0) tied (ETF and elog / emlp input)",
        tahoe_hvg, tahoe_hvg, sc_hvg, sc_hvg),
    ("hvg1024_vs_top1024", "Model-matched: expression distance on 1,024 HVGs vs Kendall on top 1,024 detected genes (count > 0)",
        tahoe_hvg, tahoe_top, sc_hvg, sc_top)]
for (tag, subtitle, pb_e, pb_r, sc_e, sc_r) in gene_sets
    suffix = tag == "all" ? "" : "_$tag"
    pb_sc_hexbin(pb_r["kendall"], pb_e["cosine"], sc_r["kendall"], sc_e["cosine"];
        xlabel="Kendall tau distance", ylabel="Cosine distance", subtitle=subtitle,
        # path="$fig_cos_dir/ts_cosine_kendall$(suffix)_100K.png")
        path="$fig_cos_dir/ts_cosine_kendall$(suffix)_$(pairs_tag).png")
    pb_sc_hexbin(pb_r["kendall"], pb_e["euclidean"], sc_r["kendall"], sc_e["euclidean"];
        xlabel="Kendall tau distance", ylabel="Euclidean distance", subtitle=subtitle,
        # path="$fig_euc_dir/ts_euclid_kendall$(suffix)_100K.png")
        path="$fig_euc_dir/ts_euclid_kendall$(suffix)_$(pairs_tag).png")
end


# histograms per metric

begin
    fig2 = Figure(size=(700, 750))

    # ax_k = Axis(fig2[1, 1], xlabel="kendall tau distance", ylabel="density", title="kendall tau distance")
    # hist!(ax_k, Float64.(lincs["kendall"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS (n=$(length(lincs["kendall"])))")
    ax_k = Axis(fig2[1, 1], xlabel="Kendall tau distance", ylabel="Density", title="Kendall tau distance")
    hist!(ax_k, Float64.(tahoe["kendall"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB (n=$(length(tahoe["kendall"])))")
    hist!(ax_k, Float64.(sc["kendall"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC (n=$(length(sc["kendall"])))")
    axislegend(ax_k, position=:lt)

    # ax_c = Axis(fig2[2, 1], xlabel="cosine distance", ylabel="density", title="cosine distance")
    # hist!(ax_c, Float64.(lincs["cosine"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS")
    ax_c = Axis(fig2[2, 1], xlabel="Cosine distance", ylabel="Density", title="Cosine distance")
    hist!(ax_c, Float64.(tahoe["cosine"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB")
    hist!(ax_c, Float64.(sc["cosine"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC")
    axislegend(ax_c, position=:ct)

    # ax_e = Axis(fig2[3, 1], xlabel="euclidean distance", ylabel="density", title="euclidean distance")
    # hist!(ax_e, Float64.(lincs_euc["euclid"]), bins=100, normalization=:pdf, color=(:orange, 0.4), label="LINCS")
    ax_e = Axis(fig2[3, 1], xlabel="Euclidean distance", ylabel="Density", title="Euclidean distance")
    hist!(ax_e, Float64.(tahoe_euc["euclid"]), bins=100, normalization=:pdf, color=(:teal, 0.4), label="Tahoe PB")
    hist!(ax_e, Float64.(sc_euc["euclidean"]), bins=100, normalization=:pdf, color=(:purple, 0.4), label="Tahoe SC")
    axislegend(ax_e, position=:ct)

    # save("$fig_dir/compare_histograms_100K.png", fig2)
    save("$fig_hist_dir/compare_histograms_$(pairs_tag).png", fig2)
    # display(fig2)
end


# entropy comparison

# lincs_ent = load("results/lincs/data/entropies/ranked_lincs_entropies.jld2")
# lincs_ent = load("results/lincs/data/entropies/lincs_ranked_entropies.jld2")
# tahoe_ent = load("results/tahoe/pb/data/entropies/ranked_pseudobulks_a10k_entropies.jld2")
tahoe_ent = load("results/tahoe/pb/data/entropies/pb_ranked_entropies.jld2")

# lincs_H = lincs_ent["entropies"]
tahoe_H = tahoe_ent["entropies"]

# entropy by rank

# begin
    # fig_rank = Figure(size=(700, 400))

    # ax = Axis(fig_rank[1, 1],
        # xlabel="Rank (1 = highest expression)",
        # ylabel="Shannon entropy (bits)")

    # scatter!(ax, 1:length(tahoe_H), Float64.(tahoe_H),
        # markersize=3, alpha=0.5, color=:teal, label="Tahoe ($(length(tahoe_H)) genes)")
    # scatter!(ax, 1:length(lincs_H), Float64.(lincs_H),
        # markersize=3, alpha=0.5, color=:orange, label="LINCS ($(length(lincs_H)) genes)")
    # axislegend(ax, position=:rb)

    # display(fig_rank)
# end


# entropy/sparsity

tahoe_spar = load("results/tahoe/pb/data/entropies/pb_ranked_sparsities.jld2")
sc_ent = load("results/tahoe/sc/data/entropies/ranked_sc_entropies.jld2")
sc_spar = load("results/tahoe/sc/data/entropies/ranked_sc_sparsities.jld2")

pb_S = tahoe_spar["sparsities"]
sc_H = sc_ent["entropies"]
sc_S = sc_spar["sparsities"]

n_genes = length(tahoe_H)

# begin
    # fig_overlay = Figure(size=(700, 500))

    # ax_ent = Axis(fig_overlay[1, 1],
        # xlabel="Rank (1 = highest expression)",
        # ylabel="Shannon entropy (bits)",
        # yaxisposition=:left,
        # xtickformat=values -> [string(Int(round(v))) for v in values],
        # title="Tahoe entropy & sparsity per rank position (300 parquets sampled for SC)")
    # ax_spar = Axis(fig_overlay[1, 1],
        # ylabel="Sparsity (1 = always 0)",
        # yaxisposition=:right)
    # hidespines!(ax_spar)
    # hidexdecorations!(ax_spar)

    # entropy scatter
    # scatter!(ax_ent, 1:n_genes, Float64.(tahoe_H),
        # markersize=4, alpha=0.5, color=Makie.wong_colors()[1])
    # scatter!(ax_ent, 1:n_genes, Float64.(sc_H),
    #     markersize=4, alpha=0.5, color=Makie.wong_colors()[2])
    # sc entropies stop at the deepest rank any cell reaches (no detected gene beyond it)
    # scatter!(ax_ent, 1:length(sc_H), Float64.(sc_H),
        # markersize=4, alpha=0.5, color=Makie.wong_colors()[2])

    # sparsity lines
    # lines!(ax_spar, 1:n_genes, Float64.(pb_S),
        # linewidth=3, linestyle=:dash, color=Makie.wong_colors()[1])
    # lines!(ax_spar, 1:n_genes, Float64.(sc_S),
    #     linewidth=3, linestyle=:dash, color=Makie.wong_colors()[2])
    # beyond the deepest populated rank every cell is undetected: sparsity 1
    # lines!(ax_spar, 1:n_genes, vcat(Float64.(sc_S), ones(n_genes - length(sc_S))),
        # linewidth=3, linestyle=:dash, color=Makie.wong_colors()[2])
    # linkxaxes!(ax_ent, ax_spar)

    # legend
    # Legend(fig_overlay[0, 1],
        # [MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=8),
         # MarkerElement(color=Makie.wong_colors()[2], marker=:circle, markersize=8),
         # MarkerElement(color=:gray50, marker=:circle, markersize=8),
         # LineElement(color=:gray50, linestyle=:dash, linewidth=2)],
        # ["Pseudo-bulk", "Single cell", "Entropy", "Sparsity"],
        # orientation=:horizontal, tellwidth=false, tellheight=true)

    # display(fig_overlay)
# end
# save("$fig_ent_dir/pb_vs_sc_entropy_sparsity.png", fig_overlay)

# entropy + sparsity per rank, pseudobulk vs single cell. detected = count > 0
#   detected only (RTF view): H over the samples that have a detected gene at that rank; undefined where none do
#   undetected as one symbol (rlog / rmlp view, where every undetected gene gets the same value 0):
#     H = (1 - s) * H_detected + H_b(s), s = sparsity, H_b = binary entropy; goes to 0 where every sample is undetected
n_pb = maximum(tahoe_ent["n_at_rank"]); n_sc = maximum(sc_ent["n_at_rank"])
fmt(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")
binary_H(s) = (s <= 0 || s >= 1) ? 0.0 : -s * log2(s) - (1 - s) * log2(1 - s)
pad(v, n, val) = vcat(Float64.(v), fill(val, n - length(v)))
function one_symbol_H(H, S, n)
    H = pad(H, n, NaN); S = pad(S, n, 1.0)
    return [S[r] >= 1 ? 0.0 : (1 - S[r]) * (isnan(H[r]) ? 0.0 : H[r]) + binary_H(S[r]) for r in 1:n]
end

function entropy_sparsity_fig(pb_H, sc_H, pb_S, sc_S; title, subtitle, path)
    # fig = Figure(size=(760, 560))
    # ax_ent = Axis(fig[1, 1],
        # xlabel="Rank (1 = highest expression)",
        # ylabel="Shannon entropy (bits)",
        # yaxisposition=:left,
        # xtickformat=values -> [string(Int(round(v))) for v in values],
        # title=title, subtitle=subtitle, subtitlesize=11, subtitlecolor=:gray30)
    # ax_spar = Axis(fig[1, 1],
    # info block in its own row above the legend (can be cropped off for slides)
    fig = Figure(size=(760, 640))
    Label(fig[0, 1], subtitle, fontsize=11, color=:gray30, tellwidth=false, justification=:center)
    ax_ent = Axis(fig[2, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Shannon entropy (bits)",
        yaxisposition=:left,
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title=title)
    ax_spar = Axis(fig[2, 1],
        ylabel="Sparsity (fraction of samples undetected)",
        yaxisposition=:right)
    hidespines!(ax_spar)
    hidexdecorations!(ax_spar)

    # lines!(ax_ent, 1:length(pb_H), Float64.(pb_H), linewidth=2, color=Makie.wong_colors()[1])
    # lines!(ax_ent, 1:length(sc_H), Float64.(sc_H), linewidth=2, color=Makie.wong_colors()[2])
    scatter!(ax_ent, 1:length(pb_H), Float64.(pb_H), markersize=4, alpha=0.5, color=Makie.wong_colors()[1])
    scatter!(ax_ent, 1:length(sc_H), Float64.(sc_H), markersize=4, alpha=0.5, color=Makie.wong_colors()[2])
    lines!(ax_spar, 1:n_genes, pad(pb_S, n_genes, 1.0), linewidth=3, linestyle=:dash, color=Makie.wong_colors()[1])
    lines!(ax_spar, 1:n_genes, pad(sc_S, n_genes, 1.0), linewidth=3, linestyle=:dash, color=Makie.wong_colors()[2])
    linkxaxes!(ax_ent, ax_spar)
    xlims!(ax_ent, 0, n_genes)

    # Legend(fig[0, 1],
        # [LineElement(color=Makie.wong_colors()[1], linewidth=3),
         # LineElement(color=Makie.wong_colors()[2], linewidth=3),
         # LineElement(color=:gray50, linewidth=2),
         # LineElement(color=:gray50, linestyle=:dash, linewidth=2)],
    Legend(fig[1, 1],
        [PolyElement(color=Makie.wong_colors()[1]),
         PolyElement(color=Makie.wong_colors()[2]),
         MarkerElement(color=:gray50, marker=:circle, markersize=8),
         LineElement(color=:gray50, linestyle=:dash, linewidth=2)],
        ["Pseudobulk", "Single cell", "Entropy", "Sparsity"],
        orientation=:horizontal, tellwidth=false, tellheight=true)
    save(path, fig, px_per_unit=2)
end

sample_note = "Pseudobulk: $(fmt(n_pb)) samples. Single cell: 300 parquets sampled ($(fmt(n_sc)) cells). Detected = count > 0."
entropy_sparsity_fig(tahoe_H, sc_H, pb_S, sc_S;
    title="Entropy and sparsity per rank: detected genes only (RTF input)",
    subtitle="Entropy over samples with a detected gene at that rank; undefined where none have one.\n" *
             "Maximum entropy = log2(n), n = min(samples with a detected gene at that rank, $(fmt(n_genes)) genes) ≤ $(round(log2(n_genes), digits=1)) bits.\n" * sample_note,
    path="$fig_ent_dir/pb_vs_sc_entropy_sparsity.png")
entropy_sparsity_fig(one_symbol_H(tahoe_H, pb_S, n_genes), one_symbol_H(sc_H, sc_S, n_genes), pb_S, sc_S;
    title="Entropy and sparsity per rank: undetected genes as one value (rlog / rmlp input)",
    subtitle="Undetected genes share one symbol (value 0): H = (1 − s)·H_detected + H_b(s), s = sparsity; 0 where all samples are undetected.\n" *
             "Maximum entropy = log2(n), n = $(fmt(n_genes)) genes + 1 undetected symbol = $(round(log2(n_genes + 1), digits=1)) bits.\n" * sample_note,
    path="$fig_ent_dir/pb_vs_sc_entropy_sparsity_undetected0.png")


# unique count diversity (PB vs SC)

pb_ud = load("results/tahoe/pb/data/entropies/pb_ranked_unique_diversity.jld2")
sc_ud = load("results/tahoe/sc/data/entropies/ranked_sc_unique_diversity.jld2")

pb_UD = pb_ud["unique_diversity_norm"]
sc_UD = sc_ud["unique_diversity_norm"]

begin
    fig_ud = Figure(size=(700, 500))

    ax_ud = Axis(fig_ud[1, 1],
        xlabel="Rank (1 = highest expression)",
        ylabel="Normalized unique count diversity",
        xtickformat=values -> [string(Int(round(v))) for v in values],
        title="Unique count diversity per rank (PB vs SC)")

    lines!(ax_ud, 1:length(pb_UD), Float64.(pb_UD),
        linewidth=2, color=Makie.wong_colors()[1], label="Pseudo-bulk")
    lines!(ax_ud, 1:length(sc_UD), Float64.(sc_UD),
        linewidth=2, color=Makie.wong_colors()[2], label="Single cell")
    axislegend(ax_ud, position=:rt)

    # display(fig_ud)
end
save("$fig_ent_dir/pb_vs_sc_unique_diversity.png", fig_ud)
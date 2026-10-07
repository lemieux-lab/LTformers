# model-free rank -> expression reconstruction: how much magnitude information do ranks still carry?
# for each gene, a lookup table maps the gene's rank in a sample to that gene's mean expression at that rank in the
# training samples; the test samples' ranks are then turned back into expression with no learning
#   ranking = the training rule: x / per-gene nonzero median (data/*_gene_medians.jld2), detected genes first,
#             undetected genes tie last (-> reconstructed as 0, exactly right)
#   full    : every gene's rank position (rlog / rmlp _full input), binned into N_BINS bins per gene
#   top1024 : position within each sample's top-1024 detected genes (RTF input); genes outside the top-1024 map to the
#             gene's mean expression over training samples where it was outside (LINCS: 978 genes, full only)
#   metrics on held-out samples (expression in the stored log space, no z-scoring):
#     per-gene r       : pearson across test samples, per gene (does the rank recover between-sample variation?)
#     per-gene r (det) : same, only over test samples where the gene is detected (PB / SC; zeros are trivially exact)
#     R2 vs gene mean  : 1 - SSE / sum((x - train gene mean)^2), i.e. variance explained beyond each gene's mean
#     per-sample r (c) : pearson across genes after subtracting train gene means (sample-specific pattern)
#   splits: LINCS by detection plate, tahoe PB random, SC by well (sample); bad LINCS plate excluded
#
# julia -t 32 scripts/analyses/rank_to_expr.jl lincs | tpb        julia -t 32,0 scripts/analyses/rank_to_expr.jl sc
# cpu_sbatch L_rank2expr   "cd /home/golem/scratch/chans/transformers && julia -t 32 scripts/analyses/rank_to_expr.jl lincs"
# cpu_sbatch TPB_rank2expr "cd /home/golem/scratch/chans/transformers && julia -t 32 scripts/analyses/rank_to_expr.jl tpb"
# cpu_sbatch TSC_rank2expr "cd /home/golem/scratch/chans/transformers && R2E_NCELLS=100000 julia -t 32,0 scripts/analyses/rank_to_expr.jl sc"

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../..", arch_dir))
using JLD2, Statistics, StatsBase, CairoMakie, Random, DataFrames, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))
using Preprocess: rank_genes, rank_features

logln(xs...) = (println(xs...); flush(stdout))
dataset = isempty(ARGS) ? "lincs" : ARGS[1]
dataset in ("lincs", "tpb", "sc") || error("dataset must be lincs, tpb or sc")
dataset == "sc" && include(joinpath(@__DIR__, "sandbox/load_sc_cells.jl"))
const N_BINS = parse(Int, get(ENV, "R2E_NBINS", "100"))
const TOP_K = 1024
rng = MersenneTwister(42)

# ---------------- data ----------------
if dataset == "lincs"
    d = load("data/lincs/lincs_trt_data.jld2")["filtered_data"]
    keep = findall(string.(d.inst.det_plate) .!= "REP.A010_JURKAT_24H_X3_B32")
    n_use = parse(Int, get(ENV, "R2E_NSAMPLES", "300000"))
    use = sort(keep[randperm(rng, length(keep))[1:min(n_use, end)]])
    X = Float32.(d.expr[:, use]); groups = string.(d.inst.det_plate[use]); d = nothing
    meds = Float32.(load("data/lincs/gene_medians.jld2", "medians"))
    out_root = "results/lincs/analyses/rank_to_expr"; ds_name = "LINCS"
elseif dataset == "tpb"
    df = load("data/tahoe/filtered_pseudobulks_alpha_10000.jld2")["df"]
    X = reduce(hcat, [Float32.(e) for e in df.expr]); groups = string.(1:size(X, 2))   # random split
    meds = Float32.(load("data/tahoe/pb_gene_medians.jld2", "medians"))
    out_root = "results/tahoe/pb/analyses/rank_to_expr"; ds_name = "Tahoe pseudobulk"
else
    X, meta = load_sc_cells(parse(Int, get(ENV, "R2E_NCELLS", "100000")))
    groups = meta.sample                                                                  # split by well
    meds = Float32.(load("data/tahoe/sc_gene_medians.jld2", "medians"))
    out_root = "results/tahoe/sc/analyses/rank_to_expr"; ds_name = "Tahoe single cell"
end
fig_dir = "$out_root/figures"; data_dir = "$out_root/data"; mkpath(fig_dir); mkpath(data_dir)
n_genes, N = size(X)
@assert length(meds) == n_genes
ug = shuffle(rng, unique(groups)); test_g = Set(ug[1:max(1, round(Int, 0.2 * length(ug)))])
te = findall(in(test_g), groups); tr = findall(g -> !(g in test_g), groups)
logln("$ds_name: $n_genes genes × $N samples; train $(length(tr)), test $(length(te)); bins $N_BINS; out $out_root")
do_top = n_genes > TOP_K

# rank position of every gene in a sample (detected genes 1..n_det by x / median; undetected = 0)
function positions!(pos::Vector{Int32}, x::AbstractVector{Float32})
    det = findall(>(0f0), x)
    order = det[sortperm(view(x, det) ./ view(meds, det), rev=true)]
    fill!(pos, 0); pos[order] .= 1:length(order)
    return length(order)
end
bin_full(p) = p == 0 ? 0 : clamp(ceil(Int, N_BINS * p / n_genes), 1, N_BINS)
bin_top(p) = (p == 0 || p > TOP_K) ? 0 : clamp(ceil(Int, N_BINS * p / TOP_K), 1, N_BINS)

# ---------------- lookup tables from the training samples ----------------
nt = Threads.nthreads()
Sf = [zeros(n_genes, N_BINS) for _ in 1:nt]; Cf = [zeros(Int, n_genes, N_BINS) for _ in 1:nt]
St = [zeros(n_genes, N_BINS) for _ in 1:nt]; Ct = [zeros(Int, n_genes, N_BINS) for _ in 1:nt]
Sa = [zeros(n_genes) for _ in 1:nt]; Ca = [zeros(Int, n_genes) for _ in 1:nt]        # outside the top-1024
Sm = [zeros(n_genes) for _ in 1:nt]                                                   # gene means
chunks = collect(Iterators.partition(tr, cld(length(tr), nt)))
Threads.@threads for c in eachindex(chunks)
    t = c; pos = Vector{Int32}(undef, n_genes)
    for j in chunks[c]
        x = view(X, :, j); positions!(pos, x)
        @inbounds for g in 1:n_genes
            v = x[g]; Sm[t][g] += v
            b = bin_full(pos[g]); b > 0 && (Sf[t][g, b] += v; Cf[t][g, b] += 1)
            if do_top
                bt = bin_top(pos[g])
                bt > 0 ? (St[t][g, bt] += v; Ct[t][g, bt] += 1) : (Sa[t][g] += v; Ca[t][g] += 1)
            end
        end
    end
end
μ = sum(Sm) ./ length(tr)
# bin means; empty bins take the nearest non-empty bin of the same gene (else the gene mean)
function table(S, C)
    T = fill(NaN, n_genes, N_BINS)
    for g in 1:n_genes
        ok = findall(>(0), view(C, g, :))
        isempty(ok) && (T[g, :] .= μ[g]; continue)
        for b in 1:N_BINS
            k = ok[argmin(abs.(ok .- b))]; T[g, b] = S[g, k] / C[g, k]
        end
    end
    return T
end
T_full = table(sum(Sf), sum(Cf))
T_top = do_top ? table(sum(St), sum(Ct)) : nothing
abs_mean = do_top ? [c > 0 ? s / c : μ[g] for (g, (s, c)) in enumerate(zip(sum(Sa), sum(Ca)))] : nothing
logln("lookup tables built")

# ---------------- reconstruct the test samples ----------------
Xt = X[:, te]
recon(kind) = begin
    R = zeros(Float32, n_genes, length(te))
    Threads.@threads for i in eachindex(te)
        pos = Vector{Int32}(undef, n_genes); positions!(pos, view(Xt, :, i))
        for g in 1:n_genes
            R[g, i] = kind == :full ? (pos[g] == 0 ? 0f0 : T_full[g, bin_full(pos[g])]) :
                      (bin_top(pos[g]) > 0 ? T_top[g, bin_top(pos[g])] : abs_mean[g])
        end
    end
    R
end
kinds = do_top ? [:full, :top1024] : [:full]
R = Dict(k => recon(k) for k in kinds)

safe_cor(a, b) = (std(a) > 0 && std(b) > 0) ? cor(a, b) : NaN
function metrics(Xh)
    pg = [safe_cor(view(Xh, g, :), view(Xt, g, :)) for g in 1:n_genes]
    pgd = [(m = view(Xt, g, :) .> 0; count(m) >= 20 ? safe_cor(view(Xh, g, m), view(Xt, g, m)) : NaN) for g in 1:n_genes]
    r2 = 1 - sum(abs2, Xh .- Xt) / sum(abs2, Xt .- μ)
    ps = [safe_cor(view(Xh, :, i) .- μ, view(Xt, :, i) .- μ) for i in axes(Xt, 2)]
    return (; per_gene_r=pg, per_gene_r_det=pgd, r2_vs_mean=r2, per_sample_rc=ps)
end
M = Dict(k => metrics(R[k]) for k in kinds)
med(v) = (w = filter(!isnan, v); isempty(w) ? NaN : median(w))
logln("\n$ds_name, $(length(te)) test samples (expression in stored log space):")
for k in kinds
    m = M[k]
    logln(@sprintf("  %-8s per-gene r median %.3f | per-gene r (detected only) median %.3f | R² vs gene mean %.3f | per-sample r (centred) median %.3f",
                   k, med(m.per_gene_r), med(m.per_gene_r_det), m.r2_vs_mean, med(m.per_sample_rc)))
end
jldsave("$data_dir/rank_to_expr.jld2"; metrics=Dict(string(k) => M[k] for k in kinds), n_bins=N_BINS, n_train=length(tr),
        n_test=length(te), dataset, gene_mean_train=μ)
logln("saved $data_dir/rank_to_expr.jld2")

# ---------------- figures ----------------
cols = Dict(:full => Makie.wong_colors()[2], :top1024 => Makie.wong_colors()[6])
labs = Dict(:full => "Full ranking", :top1024 => "Top-1024 (RTF)")
begin
    fig = Figure(size=(1500, 460), fontsize=14)
    for (p, (field, xl)) in enumerate(((:per_gene_r, "Per-gene r across test samples"), (:per_gene_r_det, "Per-gene r, detected values only"),
                                       (:per_sample_rc, "Per-sample r across genes (centred)")))
        ax = Axis(fig[1, p], xlabel=xl, ylabel=p == 1 ? "Density" : "", title=p == 1 ? ds_name : "")
        for k in kinds
            v = filter(!isnan, getfield(M[k], field)); isempty(v) && continue
            density!(ax, v, color=(cols[k], 0.35), strokecolor=cols[k], strokewidth=1.5, label="$(labs[k]) (median $(round(median(v), digits=2)))")
        end
        axislegend(ax, position=:lt, framevisible=false, labelsize=11)
    end
    Label(fig[0, :], "Expression reconstructed from ranks by a per-gene rank → mean-expression lookup ($N_BINS rank bins per gene, fit on $(length(tr)) training samples)",
          fontsize=11, color=:gray40, tellwidth=false)
    save("$fig_dir/rank_to_expr_metrics.png", fig, px_per_unit=2)
end
begin   # one test sample: true vs reconstructed, centred by the train gene mean
    i = rand(rng, eachindex(te))
    fig = Figure(size=(560 * length(kinds), 520), fontsize=14)
    for (p, k) in enumerate(kinds)
        a = Xt[:, i] .- μ; b = R[k][:, i] .- μ
        ax = Axis(fig[1, p], xlabel="True expression − gene mean", ylabel="Reconstructed − gene mean",
                  title=@sprintf("%s: r = %.2f", labs[k], safe_cor(a, b)), aspect=1)
        scatter!(ax, a, b, color=(cols[k], 0.4), markersize=3); ablines!(ax, 0, 1, color=:gray50, linestyle=:dash)
    end
    Label(fig[0, :], "$ds_name, one held-out sample", fontsize=12, tellwidth=false)
    save("$fig_dir/rank_to_expr_example_sample.png", fig, px_per_unit=2)
end
logln("done: $fig_dir")


# ---------------- shared mapping: the actual rank-feature vectors, ONE curve for all genes ----------------
# rank vectors exactly as the rank baselines get them (Preprocess.rank_genes + rank_features: (k+1-r)/k for the top-k
# detected genes, (d+1-r)/n for all genes, absent / undetected = 0). a single map f(feature value) -> mean expression,
# fit over all genes of the training samples (no gene identity), reconstructs each test expression vector
const N_VBINS = 1000
vbin(v) = v <= 0f0 ? 1 : 1 + clamp(ceil(Int, N_VBINS * v), 1, N_VBINS)     # bin 1 = absent / undetected
function rank_vectors(Xs, k)
    Xr = rank_genes(Xs, meds); nd = vec(sum(Xs .> 0f0, dims=1))
    return rank_features(Xr, nd, n_genes, k)
end
shared_kinds = do_top ? [(:full, n_genes), (:top1024, TOP_K)] : [(:full, n_genes)]
SM = Dict{Symbol,Any}(); curves = Dict{Symbol,Vector{Float64}}()
for (kind, k) in shared_kinds
    S = zeros(N_VBINS + 1); C = zeros(Int, N_VBINS + 1)
    for c in Iterators.partition(tr, 5000)                                   # fit on train, in chunks
        F = rank_vectors(X[:, c], k); Xc = view(X, :, c)
        @inbounds for i in eachindex(F); b = vbin(F[i]); S[b] += Xc[i]; C[b] += 1; end
    end
    f = [C[b] > 0 ? S[b] / C[b] : NaN for b in eachindex(S)]
    for b in eachindex(f); isnan(f[b]) && (ok = findall(>(0), C); f[b] = f[ok[argmin(abs.(ok .- b))]]); end
    curves[kind] = f
    Ft = rank_vectors(Xt, k)
    Xh = Float32.(f[vbin.(Ft)])
    pr = [safe_cor(view(Xh, :, i), view(Xt, :, i)) for i in axes(Xt, 2)]            # raw per-sample r
    m = metrics(Xh)
    SM[kind] = (; m..., per_sample_r=pr, r2_vs_global=1 - sum(abs2, Xh .- Xt) / sum(abs2, Xt .- mean(Xt)))
    logln(@sprintf("  shared %-8s per-sample r median %.3f | R² vs global mean %.3f | R² vs gene mean %.3f | per-gene r median %.3f | per-gene r (det) median %.3f",
                   kind, med(pr), SM[kind].r2_vs_global, m.r2_vs_mean, med(m.per_gene_r), med(m.per_gene_r_det)))
end
jldsave("$data_dir/rank_to_expr_shared.jld2"; metrics=Dict(string(k) => v for (k, v) in SM), curves=Dict(string(k) => v for (k, v) in curves),
        n_vbins=N_VBINS, n_train=length(tr), n_test=length(te), dataset)
begin
    fig = Figure(size=(1500, 460), fontsize=14)
    ax0 = Axis(fig[1, 1], xlabel="Rank-feature value ((k+1−r)/k; 0 = absent)", ylabel="Mean expression (shared curve)", title=ds_name)
    for (kind, _) in shared_kinds
        f = curves[kind]; lines!(ax0, [0; (1:N_VBINS) ./ N_VBINS], f, color=cols[kind], linewidth=2, label=labs[kind])
    end
    axislegend(ax0, position=:lt, framevisible=false, labelsize=11)
    for (p, (field, xl)) in enumerate(((:per_sample_r, "Per-sample r: reconstructed vs true expression"), (:per_gene_r, "Per-gene r across test samples")))
        ax = Axis(fig[1, p + 1], xlabel=xl, ylabel="Density")
        for (kind, _) in shared_kinds
            v = filter(!isnan, getfield(SM[kind], field)); isempty(v) && continue
            density!(ax, v, color=(cols[kind], 0.35), strokecolor=cols[kind], strokewidth=1.5, label="$(labs[kind]) (median $(round(median(v), digits=2)))")
        end
        axislegend(ax, position=:lt, framevisible=false, labelsize=11)
    end
    Label(fig[0, :], "Expression vectors rebuilt from the rank-feature vectors with ONE value → expression curve for all genes (no gene identity)",
          fontsize=11, color=:gray40, tellwidth=false)
    save("$fig_dir/rank_to_expr_shared.png", fig, px_per_unit=2)
end
logln("shared mapping done: $fig_dir/rank_to_expr_shared.png")

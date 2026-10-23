#=
Trivial baselines for the TSC pretraining objectives, same inputs/losses as scripts/pretrain/sc/{erecon,mlm}.jl.
erecon: train mean per HVG (etf) / per rank position (rtf); mlm: modal rank per HVG (etf) / modal gene per rank position (rtf).
Exact expectation over masked positions (masks are uniform over eligible positions), on val shards 1:2 and test shards 1:10.
Usage: julia --threads=1,0 --project=x86_64 scripts/analyses/sc_pretrain_trivial.jl [n_train_shards (default 50)]
=#

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../..", arch_dir))
using JLD2, DataFrames, Statistics, CSV, Printf, Random
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"), joinpath(@__DIR__, "../../src/tahoe"))
using Config: load_config
using LoadSC
using ProcessSC: build_batch_etf, build_batch_etf_hvg
using Preprocess: pad_token_id

const REPO = abspath(joinpath(@__DIR__, "../.."))
cd(REPO)
const N_TRAIN = isempty(ARGS) ? 50 : parse(Int, ARGS[1])
const TOP_K = 1024
const CHUNK = 2000

config = load_config("config/default.toml", Dict{String,Any}("modeltype" => "etf"); dataset="tahoe_sc")
coding_tokens, token_to_idx, n_coding = LoadSC.load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
train_shards, val_shards, test_shards = LoadSC.split_shards(LoadSC.list_shards(config["data_dir"]), 0)
hvg_idx = LoadSC.load_hvg_idx(get(config, "hvg_path", ""))
n_hvg = length(hvg_idx)
pad = pad_token_id(n_coding)
tr = shuffle(MersenneTwister(0), train_shards)[1:min(N_TRAIN, length(train_shards))]
println("TSC trivial: $(length(tr)) train shards, $n_hvg HVGs, top_k $TOP_K, n_coding $n_coding")

# per-shard chunks: (hvg expr, hvg ranks, topk ids, topk expr)
function each_chunk(f, path)
    sh = LoadSC.load_shard_pyarrow(path)
    for s in 1:CHUNK:sh.n_cells
        ci = collect(s:min(s + CHUNK - 1, sh.n_cells))
        he, hr = build_batch_etf_hvg(sh.genes_flat, sh.offsets, sh.expr_flat, ci, token_to_idx, n_coding, hvg_idx)
        ids, ex = build_batch_etf(sh.genes_flat, sh.offsets, sh.expr_flat, ci, token_to_idx, n_coding, TOP_K)
        f(he, hr, ids, ex)
    end
    GC.gc()
end

# train statistics
sum_g = zeros(Float64, n_hvg); n_cells = 0
cnt_rank = zeros(Int32, n_hvg, n_hvg)          # gene x rank
sum_r = zeros(Float64, TOP_K); cnt_r = zeros(Int64, TOP_K)
cnt_gene = zeros(Int32, n_coding, TOP_K)        # gene id x rank position
for (k, p) in enumerate(tr)
    each_chunk(p) do he, hr, ids, ex
        global n_cells += size(he, 2)
        sum_g .+= vec(sum(he, dims=2))
        @inbounds for j in axes(hr, 2), g in axes(hr, 1)
            cnt_rank[g, hr[g, j]] += 1
        end
        @inbounds for j in axes(ids, 2), r in axes(ids, 1)
            id = ids[r, j]
            (id == pad || id > n_coding || id < 1) && continue
            sum_r[r] += ex[r, j]; cnt_r[r] += 1
            cnt_gene[id, r] += 1
        end
    end
    k % 10 == 0 && println("  train $k/$(length(tr)) shards, $n_cells cells"); flush(stdout)
end
mu_g = sum_g ./ n_cells
mu_r = sum_r ./ max.(cnt_r, 1)
mode_rank = [argmax(view(cnt_rank, g, :)) for g in 1:n_hvg]
mode_gene = [argmax(view(cnt_gene, :, r)) for r in 1:TOP_K]
println("train: $n_cells cells; valid rank positions per cell (mean) $(round(sum(cnt_r) / n_cells, digits=1))")

# eval: exact expectation over eligible (maskable) positions
function evaluate(paths)
    se_e = 0.0; n_e = 0                         # erecon etf: all HVG positions
    ce_m = 0.0; acc_m = 0; n_m = 0              # mlm etf: all HVG positions, 1024 classes
    se_r = 0.0; n_r = 0                         # erecon rtf: non-PAD rank positions
    ce_g = 0.0; acc_g = 0; n_g = 0              # mlm rtf: non-PAD rank positions, n_coding classes
    for p in paths
        each_chunk(p) do he, hr, ids, ex
            @inbounds for j in axes(he, 2), g in axes(he, 1)
                se_e += (he[g, j] - mu_g[g])^2; n_e += 1
                r = hr[g, j]
                ce_m -= log((cnt_rank[g, r] + 1) / (n_cells + n_hvg)); acc_m += (r == mode_rank[g]); n_m += 1
            end
            @inbounds for j in axes(ids, 2), r in axes(ids, 1)
                id = ids[r, j]
                (id == pad || id > n_coding || id < 1) && continue
                se_r += (ex[r, j] - mu_r[r])^2; n_r += 1
                ce_g -= log((cnt_gene[id, r] + 1) / (cnt_r[r] + n_coding)); acc_g += (id == mode_gene[r]); n_g += 1
            end
        end
    end
    return (; erecon_etf=se_e / n_e, mlm_etf=ce_m / n_m, mlm_etf_acc=acc_m / n_m,
              erecon_rtf=se_r / n_r, mlm_rtf=ce_g / n_g, mlm_rtf_acc=acc_g / n_g)
end

rows = DataFrame()
const SMOKE = get(ENV, "SMOKE", "0") == "1"  # 1 val + 1 test shard
for (set, paths) in (("val", val_shards[1:(SMOKE ? 1 : 2)]), ("test", test_shards[1:(SMOKE ? 1 : 10)]))
    r = evaluate(paths)
    for (obj, arch, pred, metric, val, acc) in (
            ("erecon", "etf", "train mean per HVG", "mse", r.erecon_etf, NaN),
            ("mlm", "etf", "train modal rank per HVG", "ce", r.mlm_etf, r.mlm_etf_acc),
            ("erecon", "rtf", "train mean at rank position", "mse", r.erecon_rtf, NaN),
            ("mlm", "rtf", "train modal gene per rank position", "ce", r.mlm_rtf, r.mlm_rtf_acc))
        push!(rows, (; dataset="tahoe_sc", eval_set=set, objective=obj, arch, predictor=pred, metric, value=val, accuracy=acc,
                       n_train_shards=length(tr), n_train_cells=n_cells))
        @printf("%-5s %-7s %-4s %-36s %-3s %.5f  acc=%.4f\n", set, obj, arch, pred, metric, val, acc)
    end
end
out = joinpath(REPO, "results", "tahoe", "sc", "analyses"); mkpath(out)
CSV.write(joinpath(out, "pretrain_trivial.csv"), rows)
println("wrote $(joinpath(out, "pretrain_trivial.csv"))")

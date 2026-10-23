#=
Pretrained checkpoint check, no finetuning: for each candidate checkpoint of one (dataset, input type), pooled embeddings of a
fixed train/val sample -> label-free health (RankMe effective rank, median dim-sd) + frozen readouts on val only
(kNN k=20 cosine, linear probe x3 seeds), lvl1 (cell line) and lvl2 (drug / compound) labels, same splits as the finetunes.
Candidates: lrecon best/ (+ best ema), final student, final teacher (ema); mlm/erecon best/ vs final (run root); random init.
Usage: julia --threads=1,0 scripts/analyses/pt_ckpt_check.jl <tpb|tsc|lincs> <rtf|etf>   -> results/analyses/pt_ckpt_check/<ds>_<mt>.csv
       ... <ds> <mt> <tag> <task,label,dir> ...   explicit candidates instead of the toml -> <ds>_<mt>_<tag>.csv
=#

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../..", arch_dir))
using JLD2, CUDA, Flux, Optimisers, Random, Statistics, LinearAlgebra, DataFrames, CSV, TOML, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"), joinpath(@__DIR__, "../../src/tahoe"))
using Models, Train, Log, Args, Config, ProcessLabels, Preprocess, Extract, FTModels
using LoadSC, ProcessSC, SCTrain

const REPO = abspath(joinpath(@__DIR__, "../.."))
cd(REPO)
const DS, MT = ARGS[1], ARGS[2]
const TAG = length(ARGS) > 2 ? ARGS[3] : ""
const EXTRA = [Tuple(split(a, ",")) for a in ARGS[4:end]]
const N_TRAIN, N_VAL, K = 50_000, 20_000, 20
fmt = DS == "tpb" ? "tahoe" : DS == "tsc" ? "tahoe_sc" : "lincs"
dskey = DS == "tpb" ? "tahoe_pb" : DS == "tsc" ? "tahoe_sc" : "lincs"
CUDA.device!(0)

# ---- candidates from config/pretrained_models.toml (run root = entry minus best/final/ema)
toml = TOML.parsefile("config/pretrained_models.toml")[dskey]
has_state(p) = isfile(joinpath(p, "model_state.jld2")) && isfile(joinpath(p, "model_config.jld2"))
cands = Tuple{String, String, String}[]  # (task, ckpt, dir)
for task in ("mlm", "erecon", "lrecon")
    isempty(EXTRA) || break
    p = rstrip(get(get(toml, task, Dict()), MT, ""), '/')
    p == "" && continue
    # run = basename(p) in ("best", "final", "ema") ? dirname(p) : p
    run = p
    while basename(run) in ("best", "final", "ema"); run = dirname(run); end
    cs = task == "lrecon" ?
        [("best", joinpath(run, "best")), ("best_ema", joinpath(run, "best", "ema")),
         ("final", isdir(joinpath(run, "final")) ? joinpath(run, "final") : run),
         ("final_ema", isdir(joinpath(run, "final", "ema")) ? joinpath(run, "final", "ema") : joinpath(run, "ema"))] :
        # [("best", joinpath(run, "best")), ("final", run)]
        [("best", joinpath(run, "best")), ("final", isdir(joinpath(run, "final")) ? joinpath(run, "final") : run)]
    for (ck, dir) in cs
        has_state(dir) ? push!(cands, (task, ck, dir)) : println("skip $task $ck: no checkpoint at $dir")
    end
end
for (task, ck, dir) in EXTRA
    has_state(dir) ? push!(cands, (String(task), String(ck), String(dir))) : error("no checkpoint at $dir")
end
isempty(cands) && error("no checkpoints for $dskey $MT")
println("candidates:"); foreach(c -> println("  ", c), cands)

# ---- data: (level => (Xtr, ytr, Xva, yva, n_cls, balanced)), features as the finetunes feed them
to_x(x) = MT == "rtf" ? cu(Int32.(x)) : cu(Float32.(x))
rng = MersenneTwister(1)
function balanced_pick(labels, n; balanced)
    if !balanced
        return length(labels) <= n ? collect(1:length(labels)) : randperm(rng, length(labels))[1:n]
    end
    by = Dict{Int, Vector{Int}}()
    for (j, l) in enumerate(labels); push!(get!(by, l, Int[]), j); end
    q = cld(n, length(by))
    return reduce(vcat, [length(v) <= q ? v : shuffle(rng, v)[1:q] for v in values(by)])
end

levels = Dict{String, Any}()
if fmt != "tahoe_sc"
    empty!(ARGS); append!(ARGS, ["-t", MT, "--task", "mlm", "-m", "emb", "-l", "lvl2", "--data_format", fmt,
                                 "-b", "100", "-e", "0", "--wandb_mode", "disabled"])
    args = load_finetune_args()
    config = load_config(args["config"], args)
    resolve_data_path!(config)
    data_key = fmt == "lincs" ? "filtered_data" : "df"
    data = load(config["data_path"])[data_key]
    data_expr = fmt == "lincs" ? (data isa Matrix{Float32} ? data : Float32.(data.expr)) : Float32.(reduce(hcat, data.expr))
    meta_df = fmt == "lincs" ? data.inst : data
    hvg_idx = nothing
    MT == "etf" && ((data_expr, hvg_idx) = select_hvg(data_expr, get(config, "n_hvg", 1024)))
    top_k = get(config, "top_k", 1024)
    for lvl in ("lvl1", "lvl2")
        c = copy(config); c["level"] = lvl
        c["group_split"] = (lvl == "lvl2" && fmt == "tahoe") ? 1 : 0  # tpb lvl2: (drug, dose) groups, as the wpt sweeps
        d = dsplit(data_expr, c; label_path=get(c, "label_path", ""),
                   label_source=(fmt == "tahoe" ? meta_df : nothing),
                   inst_df=(fmt == "lincs" && !isa(data, Matrix) ? data.inst : nothing),
                   gene_df=(fmt == "lincs" && !isa(data, Matrix) ? data.gene : nothing),
                   ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes)
        ytr, yva = Flux.onecold(d.y_train), Flux.onecold(d.y_val)
        itr = balanced_pick(ytr, N_TRAIN; balanced=(lvl == "lvl2"))
        iva = balanced_pick(yva, N_VAL; balanced=false)
        Xtr, Xva = d.X_train[:, itr], d.X_val[:, iva]
        if MT == "rtf" && top_k < d.n_genes
            Xtr, Xva = Xtr[1:top_k, :], Xva[1:top_k, :]
        end
        seq = MT == "rtf" ? min(top_k, d.n_genes) : d.n_genes
        levels[lvl] = (; Xtr, ytr=ytr[itr], Xva, yva=yva[iva], n_cls=d.n_classifications, balanced=(lvl == "lvl2"),
                       n_genes_data=d.n_genes, seq_len_data=seq, hvg_idx)
        println("$lvl: $(length(itr)) train / $(length(iva)) val samples (of $(length(ytr)) / $(length(yva))), $(d.n_classifications) classes")
    end
else
    empty!(ARGS); append!(ARGS, ["-t", MT, "--task", "mlm", "-m", "emb", "-l", "lvl2", "-b", "100", "-e", "0", "--wandb_mode", "disabled"])
    args = load_sc_finetune_args()
    config = load_config(args["config"], args; dataset="tahoe_sc")
    config["data_format"] = "tahoe_sc"
    coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
    all_shards = list_shards(config["data_dir"])
    hvg_idx = MT == "etf" ? JLD2.load(config["hvg_path"])["hvg_idx"] : nothing
    top_k = get(config, "top_k", 1024)
    # (shard, cell idx, label) list -> shard => (cell idx, labels)
    function group(cells)
        g = Dict{String, Tuple{Vector{Int}, Vector{Int}}}()
        for (sp, ci, lab) in cells
            e = get!(g, sp, (Int[], Int[])); push!(e[1], ci); push!(e[2], lab)
        end
        return g
    end
    # read model-ready features + labels for (shard => cell idx, label) picks
    function read_cells(picks, n_cls)
        xs, ls = Any[], Int[]
        for (sp, (cis, labs)) in picks
            for (x, y) in finetune_batches_from_shard(sp, cis, labs, token_to_idx, n_coding, top_k, 1000, MT, n_cls;
                                                       hvg_idx=hvg_idx, process_cell_topk_flat_fn=process_cell_topk_flat,
                                                       cell_to_dense_flat_fn=cell_to_dense_flat!)
                push!(xs, x); append!(ls, Flux.onecold(y))
            end
        end
        return reduce(hcat, xs), ls
    end
    for (lvl, split, sub, n_tr_shards) in (("lvl1", "well_cl", 100, 30), ("lvl2", "drug_dose", 0, 150))
        c = copy(config); c["level"] = lvl; c["sc_split"] = split; c["ft_eval_shards"] = 40
        d = load_sc_finetune_data_streaming(all_shards, lvl, token_to_idx, n_coding, top_k, MT;
                                            pb_data_path=get(c, "pb_data_path", ""), hvg_idx=hvg_idx, subset_shards=sub, split_by=split,
                                            process_cell_topk_flat_fn=process_cell_topk_flat,
                                            cell_to_dense_flat_fn=cell_to_dense_flat!, oversmpl_fn=oversmpl,
                                            meta_dir=get(c, "meta_dir", ""), regression_pairs_fn=get_regression_pairs_pca,
                                            identity_baseline_fn=identity_baseline)
        val_paths, _ = sc_eval_paths(d, c)  # pretrain-unseen val shards, as the finetunes
        trs = shuffle(rng, sort(collect(keys(d.train_shard_map))))[1:min(n_tr_shards, length(d.train_shard_map))]
        tr = [(sp, ci, lab) for sp in trs for (ci, lab) in zip(d.train_shard_map[sp]...)]
        tr = tr[balanced_pick([t[3] for t in tr], N_TRAIN; balanced=(lvl == "lvl2"))]
        va = [(sp, ci, lab) for sp in val_paths for (ci, lab) in zip(d.val_shard_map[sp]...)]
        va = va[balanced_pick([t[3] for t in va], N_VAL; balanced=false)]
        n_cls = d.n_classifications
        Xtr, ytr = read_cells(group(tr), n_cls)
        Xva, yva = read_cells(group(va), n_cls)
        levels[lvl] = (; Xtr, ytr, Xva, yva, n_cls, balanced=(lvl == "lvl2"),
                       n_genes_data=(MT == "rtf" ? d.n_genes : n_coding), seq_len_data=(MT == "rtf" ? top_k : d.n_genes), hvg_idx)
        println("$lvl: $(length(ytr)) train cells from $(length(trs)) shards / $(length(yva)) val cells from $(length(val_paths)) shards, $n_cls classes")
        d = nothing; GC.gc(true)
    end
end

# ---- readouts
function embed(pt, X)
    parts = Matrix{Float32}[]
    for s in 1:256:size(X, 2)
        push!(parts, cpu(pool_embed(pt, to_x(X[:, s:min(s + 255, size(X, 2))]))))
    end
    return reduce(hcat, parts)
end

function knn_acc(Ztr, ytr, Zva, yva, n_cls; k=K)
    nrm(Z) = Z ./ max.(sqrt.(sum(Z .^ 2, dims=1)), 1f-8)
    Tg = cu(nrm(Ztr)); Vn = nrm(Zva); correct = 0
    for s in 1:1000:size(Vn, 2)
        e = min(s + 999, size(Vn, 2))
        S = cpu(cu(Vn[:, s:e])' * Tg)  # (chunk, n_train) cosine
        for (r, j) in enumerate(s:e)
            nb = partialsortperm(view(S, r, :), 1:k, rev=true)
            votes = zeros(Int, n_cls)
            for t in nb; votes[ytr[t]] += 1; end
            correct += argmax(votes) == yva[j]
        end
    end
    return correct / size(Vn, 2)
end

function linprobe(Ztr, ytr, Zva, yva, n_cls; seed, balanced, steps=3000, bs=256, lr=1f-3)
    Random.seed!(seed)
    m = cu(Dense(size(Ztr, 1) => n_cls)); opt = Flux.setup(Optimisers.AdamW(lr), m)
    by = Dict{Int, Vector{Int}}(); for (j, l) in enumerate(ytr); push!(get!(by, l, Int[]), j); end
    cs = collect(keys(by)); Zv = cu(Zva); best = 0.0
    for step in 1:steps
        idx = balanced ? [rand(by[rand(cs)]) for _ in 1:bs] : rand(1:length(ytr), bs)
        x = cu(Ztr[:, idx]); y = cu(Flux.onehotbatch(ytr[idx], 1:n_cls))
        g = Flux.gradient(mm -> Flux.logitcrossentropy(mm(x), y), m)
        Flux.update!(opt, m, g[1])
        step % 500 == 0 && (best = max(best, mean(Flux.onecold(cpu(m(Zv))) .== yva)))  # best over evals, as the finetunes
    end
    return best
end

function readouts(Etr, ytr, Eva, yva, n_cls; balanced)
    μ = mean(Etr, dims=2); σ = std(Etr, dims=2); σ[σ .< 1f-6] .= 1f0
    Ztr, Zva = (Etr .- μ) ./ σ, (Eva .- μ) ./ σ
    sv = svdvals(Eva); p = sv ./ sum(sv) .+ 1f-7
    lp = [linprobe(Ztr, ytr, Zva, yva, n_cls; seed=s, balanced) for s in 1:3]
    return (rankme=exp(-sum(p .* log.(p))), dim_sd=median(vec(std(Eva, dims=2))),
            knn=knn_acc(Ztr, ytr, Zva, yva, n_cls), lp=mean(lp), lp_sd=std(lp))
end

# random init: same architecture as the first candidate, untrained (mlm encoder)
function random_pt(dir, L)
    ac = FTModels._load_arch_config(dir, config); st = load(joinpath(dir, "model_state.jld2"))["model_state"]
    seq = size(st.pos_emb.weight, 2)
    kw = (n_genes=(MT == "rtf" ? size(st.embedding.weight, 2) - 1 : seq), embed_dim=ac["embed_dim"], n_layers=ac["n_layers"],
          n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq)
    Random.seed!(1)
    return FTModels._pt_ctor(MT, "mlm")(; kw..., n_classes=10)
end

rows = DataFrame()
for (task, ck, dir) in vcat([("random", "init", cands[1][3])], cands)
    for lvl in ("lvl1", "lvl2")
        L = levels[lvl]
        c = copy(config); c["modeltype"] = MT; c["task"] = task == "random" ? "mlm" : task; c["model_dir"] = dir
        pt = task == "random" ? random_pt(dir, L) :
             first(build_pt_model(c; n_genes_data=L.n_genes_data, seq_len_data=L.seq_len_data, hvg_idx=L.hvg_idx))
        pt = fix_gpu_dropout(cu(pt)); Flux.testmode!(pt)
        Etr, Eva = embed(pt, L.Xtr), embed(pt, L.Xva)
        r = readouts(Etr, L.ytr, Eva, L.yva, L.n_cls; balanced=L.balanced)
        push!(rows, (; dataset=DS, modeltype=MT, level=lvl, task, ckpt=ck, r..., n_train=length(L.ytr), n_val=length(L.yva), dir); cols=:union)
        @printf("%-6s %-6s %-9s %s  rankme=%6.1f  dim_sd=%8.3g  knn=%.3f  linear=%.3f±%.3f\n", lvl, task, ck, DS * "_" * MT,
                r.rankme, r.dim_sd, r.knn, r.lp, r.lp_sd); flush(stdout)
        pt = nothing; GC.gc(); CUDA.reclaim()
    end
end
mkpath("results/analyses/pt_ckpt_check")
out = "results/analyses/pt_ckpt_check/$(DS)_$(MT)$(TAG == "" ? "" : "_" * TAG).csv"
CSV.write(out, rows)
println("wrote $out")

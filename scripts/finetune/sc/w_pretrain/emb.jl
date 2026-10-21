using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, Extract, FTModels
# using LoadSC, ProcessSC

using LoadSC, ProcessSC, SCTrain

args = load_sc_finetune_args()
# config = load_config(args["config"], args)

# emb mode tag: emb (mlp probe) or emb_linear
emb_tag = something(args["probe"], "mlp") == "linear" ? "emb_linear" : "emb"
config = load_config(args["config"], args,
                     hp_section=["finetune", "w_pretrain", args["modeltype"], wpt_task_tag(args), args["level"], emb_tag],
                     dataset="tahoe_sc")
config["data_format"] = "tahoe_sc"
resolve_model_dir!(config)
Extract.FINAL_LN[] = something(get(config, "final_ln", nothing), 0) == 1
resolve_lvl3_cells!(config)

# seed
seed = get(config, "seed", nothing)
if !isnothing(seed)
    Random.seed!(seed)
    CUDA.seed!(seed)
    println("Random seed: $seed")
end

is_regression = config["level"] == "lvl3"
use_oversmpl = config["level"] == "lvl2" && !is_regression

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

start_time = now()
# timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))

# + pid and seed so same-minute runs don't share a save_dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", "0") * "_p" * string(getpid()) *
            (isnothing(get(config, "seed", nothing)) ? "" : "_s$(config["seed"])")
# --sc_split
sc_split = something(get(config, "sc_split", nothing), config["level"] == "lvl1" ? "well_cl" : "drug_dose")
config["sc_split"] = sc_split
sc_split == "well_cl" && (timestamp *= "_wcl")
sc_split == "well" && (timestamp *= "_well")

# data
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])

# gene selection
hvg_idx = nothing
if config["modeltype"] == "etf"
    hvg_path = get(config, "hvg_path", "")
    if hvg_path != "" && isfile(hvg_path)
        hvg_data = JLD2.load(hvg_path)
        hvg_idx = hvg_data["hvg_idx"]
        println("loaded $(length(hvg_idx)) HVG indices")
    end
end

top_k = get(config, "top_k", 1024)
# PB data for per-cell lvl3
sc_lvl3_percell = !get(config, "sc_lvl3_pseudobulk", false)
pb_expr_for_percell = nothing
pb_df_for_percell = nothing
if sc_lvl3_percell && config["level"] == "lvl3"
    pb_path = get(config, "pb_data_path", "")
    pb_path == "" && error("sc_lvl3_percell requires pb_data_path in config")
    println("loading PB data for per-cell SC lvl3 targets: $pb_path")
    pb_data = load(pb_path)["df"]
    pb_expr_for_percell = Float32.(reduce(hcat, pb_data.expr))
    pb_df_for_percell = pb_data
end

d = load_sc_finetune_data_streaming(all_shards, config["level"], token_to_idx, n_coding, top_k, config["modeltype"];
                           pb_data_path=get(config, "pb_data_path", ""),
                           hvg_idx=hvg_idx,
                           subset_shards=get(config, "subset_shards", 0),
                           split_by=sc_split,
                           process_cell_topk_flat_fn=process_cell_topk_flat,
                           cell_to_dense_flat_fn=cell_to_dense_flat!,
                           oversmpl_fn=oversmpl,
                           source_cell=get(config, "source_cell", ""),
                           target_cell=get(config, "target_cell", ""),
                           dose=get(config, "dose", ""),
                           meta_dir=get(config, "meta_dir", ""),
                           regression_pairs_fn=get_regression_pairs_pca,
                           sc_lvl3_percell=sc_lvl3_percell,
                           pb_expr=pb_expr_for_percell,
                           pb_df=pb_df_for_percell,
                           identity_baseline_fn=identity_baseline)
is_streaming = d.train_shard_map !== nothing

id_baseline = is_regression ? d.id_baseline : nothing


if config["modeltype"] == "rtf"
    # RTF: gene ids, top_k
    seq_len = top_k
    n_genes_for_model = d.n_genes
else
    # ETF: expression, HVG applied
    seq_len = d.n_genes
    n_genes_for_model = n_coding
end

if is_streaming
    # stream embeddings shard-by-shard, then train MLP

    # # 1) pretrained encoder
    # ac = FTModels._load_arch_config(config["model_dir"], config)
    # state = JLD2.load("$(config["model_dir"])/model_state.jld2")["model_state"]

    # 1) pretrained encoder (dims from checkpoint)
    pt_model, ac = build_pt_model(config; n_genes_data=n_genes_for_model, seq_len_data=seq_len, hvg_idx=hvg_idx)

    # embedding fn per task
    if config["modeltype"] == "etf"
        embed_fn = get_embeds_exp
    elseif config["task"] in ("lrecon", "erecon")
        embed_fn = get_embeds_lrecon
    else
        embed_fn = get_embeds
    end

    # # load pretrained model
    # if config["modeltype"] == "etf" && config["task"] == "lrecon"
        # pt_model = ExpLReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            # hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # elseif config["modeltype"] == "etf" && config["task"] == "erecon"
        # pt_model = ExpEReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            # hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # elseif config["modeltype"] == "etf" && config["task"] == "mlm"
        # pt_model = ExpModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_classes=n_genes_for_model,
            # n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            # dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # elseif config["modeltype"] == "rtf" && config["task"] == "lrecon"
        # pt_model = RankLReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            # hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # elseif config["modeltype"] == "rtf" && config["task"] == "erecon"
        # pt_model = RankEReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            # hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # else
        # pt_model = RankModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            # n_layers=ac["n_layers"], n_classes=n_genes_for_model,
            # n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            # dropout_prob=ac["drop_prob"], seq_len=seq_len)
    # end
    # Flux.loadmodel!(pt_model, FTModels._clean_state(state, pt_model))

    pt_model = fix_gpu_dropout(cu(pt_model))
    Flux.testmode!(pt_model)


    embed_dim = ac["embed_dim"]

    # batch embedding
    function _encode_batch(x_batch, pt_model, modeltype, task)
        if modeltype == "etf"
            x_gpu = cu(Float32.(x_batch))
            x3d = reshape(x_gpu, 1, size(x_gpu)...)
            projected = pt_model.proj(x3d)
            gene_ids = cu(Int32.(1:size(x_batch, 1)))
            combined = projected .+ pt_model.pos_emb(gene_ids)
            dropped = pt_model.emb_dropout(combined)
            transformed = pt_model.transformer(dropped)
            emb = cpu(dropdims(mean(transformed, dims=2), dims=2))
            CUDA.unsafe_free!(x_gpu)
        else
            x_gpu = cu(Int32.(x_batch))
            # PAD-aware masked mean pool
            emb = cpu(masked_mean_pool(encode_rank(pt_model, x_gpu)...))
            CUDA.unsafe_free!(x_gpu)
        end
        return emb
    end

    # stream embeddings into preallocated matrix
    function _stream_extract_embeddings!(output_mat, output_labels, shard_paths, shard_map,
                                          pt_model, n_total, label; n_cls=d.n_classifications)
        col_offset = 0
        for (si, shard_path) in enumerate(shard_paths)
            cell_indices, cell_labels = shard_map[shard_path]
            batches = finetune_batches_from_shard(shard_path, cell_indices, cell_labels,
                                                   token_to_idx, n_coding, top_k,
                                                   config["batch_size"], config["modeltype"], n_cls;
                                                   hvg_idx=hvg_idx,
                                                   process_cell_topk_flat_fn=process_cell_topk_flat,
                                                   cell_to_dense_flat_fn=cell_to_dense_flat!)
            for (x_batch, y_batch) in batches
                bs = size(x_batch, 2)
                emb = _encode_batch(x_batch, pt_model, config["modeltype"], config["task"])
                output_mat[:, col_offset+1:col_offset+bs] .= emb
                output_labels[:, col_offset+1:col_offset+bs] .= y_batch
                col_offset += bs
            end
            if si % 50 == 0
                println("  $label embed extraction: shard $si/$(length(shard_paths)), cells=$col_offset/$n_total")
                flush(stdout)
            end
        end
        println("$label embedding extraction complete: $col_offset cells")
        return col_offset
    end

    # 2) train embeddings: same train mix as sc_finetune! (max_ft_steps x batch_size cells, class-balanced at lvl2)
    n_mix = get(config, "max_ft_steps", 0) > 0 ? config["max_ft_steps"] * config["batch_size"] : d.n_train_cells
    picked = draw_train_cells(d.train_cells, d.label_to_id, n_mix; balanced=use_oversmpl)
    mix_map = Dict{String, Tuple{Vector{Int}, Vector{Int}}}()
    for j in picked
        sp, ci, lab, _ = d.train_cells[j]
        e = get!(mix_map, sp, (Int[], Int[]))
        push!(e[1], ci); push!(e[2], d.label_to_id[lab])
    end
    n_train = length(picked)
    train_input = zeros(Float32, embed_dim, n_train)
    y_train = zeros(Float32, d.n_classifications, n_train)
    println("streaming train embedding extraction: $(n_train) cells from $(length(mix_map)) shards (balanced=$use_oversmpl)...")
    col_train = _stream_extract_embeddings!(train_input, y_train, sort(collect(keys(mix_map))), mix_map,
                                             pt_model, n_train, "train")
    if col_train < n_train
        train_input = train_input[:, 1:col_train]
        y_train = y_train[:, 1:col_train]
    end
    # mix is already class-balanced: one shuffled pass, as sc_finetune!
    use_oversmpl = false

    # # 3) val embeddings
    # n_val = d.n_val_cells
    # val_input = zeros(Float32, embed_dim, n_val)
    # y_val = zeros(Float32, d.n_classifications, n_val)
    # println("streaming val embedding extraction: $(n_val) cells...")
    # col_val = _stream_extract_embeddings!(val_input, y_val, d.val_shard_paths, d.val_shard_map,
                                           # pt_model, n_val, "val")

    # 3) val embeddings: same pretrain-unseen val shards as sc_finetune!
    val_paths, test_paths = sc_eval_paths(d, config)
    n_val = sum(length(d.val_shard_map[sp][1]) for sp in val_paths)
    val_input = zeros(Float32, embed_dim, n_val)
    y_val = zeros(Float32, d.n_classifications, n_val)
    println("streaming val embedding extraction: $(n_val) cells...")
    col_val = _stream_extract_embeddings!(val_input, y_val, val_paths, d.val_shard_map,
                                           pt_model, n_val, "val")
    if col_val < n_val
        val_input = val_input[:, 1:col_val]
        y_val = y_val[:, 1:col_val]
    end

    # # 4) test embeddings
    # n_test = d.n_test_cells
    # test_input = zeros(Float32, embed_dim, n_test)
    # y_test = zeros(Float32, d.n_classifications, n_test)
    # println("streaming test embedding extraction: $(n_test) cells...")
    # col_test = _stream_extract_embeddings!(test_input, y_test, d.test_shard_paths, d.test_shard_map,
                                            # pt_model, n_test, "test")
    # if col_test < n_test
        # test_input = test_input[:, 1:col_test]
        # y_test = y_test[:, 1:col_test]
    # end

    # # free pretrained model
    # pt_model = nothing
    # GC.gc(true)
    # CUDA.reclaim()

    # # 5) MLP head
    # ft_model = Flux.Chain(
        # Flux.Dense(embed_dim => config["hidden_dim"], gelu),
        # Flux.LayerNorm(config["hidden_dim"]),
        # Flux.Dropout(config["drop_prob"]),
        # Flux.Dense(config["hidden_dim"] => d.n_classifications))
    # ft_model = fix_gpu_dropout(cu(ft_model))
    # opt = Flux.setup(Optimisers.AdamW(config["lr"]), ft_model)

    # 4) test: streamed through encoder + head at eval time (same _eval as sc_finetune!), pt_model kept
    test_input = nothing; y_test = nothing
    GC.gc(true)
    CUDA.reclaim()

    # 5) probe head (mlp or linear) on z-scored embeddings
    emb_stats = standardize_embeds!(train_input, val_input)
    ft_model = probe_head(config, embed_dim, d.n_classifications)
    ft_model = fix_gpu_dropout(cu(ft_model))
    opt = Flux.setup(Optimisers.AdamW(config["lr"], (0.9, 0.999), Float64(something(get(config, "weight_decay", nothing), 0.0))), ft_model)
else
    # non-streaming (lvl3)
    X_train = d.X_train
    X_val, X_test = d.X_val, d.X_test
    # ft_model, train_input, val_input, test_input = build_embm(config, X_train, X_test,
                                                    # n_genes_for_model, d.n_classifications; X_val=X_val,
                                                    # seq_len=seq_len)
    # y_train = d.y_train
    # y_val = d.y_val
    # y_test = d.y_test
    # opt = Flux.setup(Optimisers.AdamW(config["lr"]), ft_model)

    ft_model, train_input, val_input, test_input = build_embm(config, X_train, X_test,
                                                    n_genes_for_model, d.n_classifications; X_val=X_val,
                                                    seq_len=seq_len, hvg_idx=hvg_idx)
    y_train = d.y_train
    y_val = d.y_val
    y_test = d.y_test
    emb_stats = standardize_embeds!(train_input, val_input, test_input)
    opt = Flux.setup(Optimisers.AdamW(config["lr"], (0.9, 0.999), Float64(something(get(config, "weight_decay", nothing), 0.0))), ft_model)
end

# save dir
dataset_tag = joinpath("tahoe", "sc")
# save_dir = joinpath("results", dataset_tag, "finetune", "w_pretrain", config["level"],
                    # config["modeltype"], config["task"], "emb", timestamp)
# mkpath(save_dir)

save_dir = joinpath("results", dataset_tag, "finetune", "w_pretrain", config["level"],
                    config["modeltype"], wpt_task_tag(config), emb_tag, timestamp)
mkpath(save_dir)
isnothing(emb_stats) || jldsave(joinpath(save_dir, "emb_stats.jld2"); mu=emb_stats[1], sd=emb_stats[2])
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
# wandb = init_wandb(config, wandb_project(config, "FT"; sc=true), "emb_sc_$(config["modeltype"])_$(config["level"])$(seed_tag)_$(timestamp)")

wandb = init_wandb(config, wandb_project(config, "FT"; sc=true), "$(config["modeltype"])_$(wpt_task_tag(config))_$(emb_tag)_sc_$(config["level"])$(seed_tag)_$(timestamp)")
wb = get(config, "wandb_mode", "disabled") != "disabled" ? wandb : nothing

# train
train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
all_preds = is_regression ? Float32[] : Int[]
all_trues = is_regression ? Float32[] : Int[]

global_step = 0
ft_step_limit = get(config, "max_ft_steps", 0)
use_max_steps = ft_step_limit > 0
done = false
# --patience: stop after this many evals without val improvement (0 = off)
patience = something(get(config, "patience", nothing), 0)
since_best = 0
best_val_loss = Inf32
best_val_acc = -Inf  # classification: best/ = highest val accuracy (val loss bottoms out early under held-out wells)
val_accs = Float64[]
best_epoch = 0
# oversampling indices from y_train
if use_oversmpl
    if is_streaming
        # class -> indices
        cidx_dict = Dict{Int, Vector{Int}}()
        for j in 1:size(y_train, 2)
            cls = argmax(y_train[:, j])
            push!(get!(cidx_dict, cls, Int[]), j)
        end
        cs = collect(keys(cidx_dict))
    else
        cidx_dict, cs = d.cidx_dict, d.cs
    end
end

# streaming: one shuffled pass over the train mix, val every ft_eval_every steps (as sc_finetune!)
stream_pass = is_streaming && use_max_steps
eval_every = something(get(config, "ft_eval_every", nothing), 4000)
eval_every = eval_every > 0 ? eval_every : ft_step_limit
mix_perm = stream_pass ? randperm(size(train_input, 2)) : nothing
n_total_epochs = if use_max_steps
    bpe = stream_pass ? eval_every : div(size(train_input, 2), config["batch_size"])
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

# # test eval for model m
# function run_test(m)

# streaming test: encoder + head over the test shards, same _eval (well / well_cl / pt-unseen) as sc_finetune!
test_ga = Dict{Symbol, Any}()
if is_streaming
    fk = (; token_to_idx, n_coding, top_k, batch_size=config["batch_size"], feat_mt=config["modeltype"],
            n_cls=d.n_classifications, hvg_idx=hvg_idx,
            to_x=(config["modeltype"] == "rtf" ? (x -> CuArray(Int32.(x))) : (x -> CuArray(x))),
            process_cell_topk_flat_fn=process_cell_topk_flat, cell_to_dense_flat_fn=cell_to_dense_flat!)
    unseen = is_regression ? nothing : sc_pt_unseen_shards(config)
    emb_mu, emb_sd = isnothing(emb_stats) ? (nothing, nothing) : (cu(emb_stats[1]), cu(emb_stats[2]))
end
function run_test(m; which=:final)
    if is_streaming
        zf = isnothing(emb_mu) ? identity : (e -> (e .- emb_mu) ./ emb_sd)
        res = SCTrain._eval(x -> m(zf(pool_embed(pt_model, x))), fk, is_regression; paths=test_paths, shard_map=d.test_shard_map,
                            label="test", group_map=(is_regression ? nothing : d.test_group_map), unseen=unseen)
        length(res) == 5 && (test_ga[which] = res[5])
        return res[1], res[3], res[4]
    end
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]
    eval_losses = Float32[]
    n_test = size(test_input, 2)
    for s in 1:config["batch_size"]:n_test
        e = min(s + config["batch_size"] - 1, n_test)
        x_gpu = cu(test_input[:, s:e])
        y_gpu = cu(y_test[:, s:e])
        logits = m(x_gpu)
        if is_regression
            push!(eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
            append!(epoch_preds, vec(cpu(logits)))
            append!(epoch_trues, vec(cpu(y_gpu)))
        else
            push!(eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            append!(epoch_preds, Flux.onecold(cpu(logits)))
            append!(epoch_trues, Flux.onecold(cpu(y_gpu)))
        end
    end
    return mean(eval_losses), epoch_preds, epoch_trues
end

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)

    # train epoch
    Flux.trainmode!(ft_model)
    epoch_losses = Float32[]
    n_train = size(train_input, 2)
    num_batches = stream_pass ? eval_every : div(n_train, config["batch_size"])
    perm = use_oversmpl ? nothing : stream_pass ? mix_perm : randperm(n_train)

    for i in 1:num_batches
        if use_oversmpl
            batch_idx = [rand(cidx_dict[rand(cs)]) for _ in 1:config["batch_size"]]
        else
            s = (stream_pass ? global_step : i - 1) * config["batch_size"] + 1
            e = min(s + config["batch_size"] - 1, n_train)
            batch_idx = perm[s:e]
        end

        x_gpu = cu(train_input[:, batch_idx])
        y_gpu = cu(y_train[:, batch_idx])

        lv, grads = Flux.withgradient(ft_model) do m
            preds = m(x_gpu)
            is_regression ? Flux.mse(preds, y_gpu) : Flux.logitcrossentropy(preds, y_gpu)
        end
        Flux.update!(opt, ft_model, grads[1])
        push!(epoch_losses, Float32(cpu(lv)))
        global global_step += 1
        if use_max_steps && global_step >= ft_step_limit
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval
    Flux.testmode!(ft_model)
    val_eval_losses = Float32[]
    val_correct = 0; val_n = 0
    n_val = size(val_input, 2)
    for s in 1:config["batch_size"]:n_val
        e = min(s + config["batch_size"] - 1, n_val)
        x_gpu = cu(val_input[:, s:e])
        y_gpu = cu(y_val[:, s:e])
        logits = ft_model(x_gpu)
        if is_regression
            push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
        else
            push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            val_correct += sum(Flux.onecold(cpu(logits)) .== Flux.onecold(cpu(y_gpu))); val_n += size(y_gpu, 2)
        end
    end
    push!(val_losses, mean(val_eval_losses))
    if patience > 0 && !is_regression
        va = val_correct / max(val_n, 1)
        if va > best_val_acc || (va == best_val_acc && val_losses[end] < best_val_loss)
            global since_best = 0
        else
            global since_best += 1
            since_best >= patience && (global done = true)
        end
    end

    # test eval (final epoch)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        final_test_loss, epoch_preds, epoch_trues = run_test(ft_model)
        push!(test_losses, final_test_loss)
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    push!(val_accs, is_regression ? NaN : val_correct / max(val_n, 1))
    # regression: lowest val loss; classification: highest val accuracy, ties -> lower val loss
    improved = is_regression ? val_losses[end] < best_val_loss :
               (val_accs[end] > best_val_acc || (val_accs[end] == best_val_acc && val_losses[end] < best_val_loss))
    if improved
        global best_val_loss = val_losses[end]
        global best_val_acc = val_accs[end]
        global best_epoch = epoch
        best_dir = joinpath(save_dir, "best")
        mkpath(best_dir)

        log_model(ft_model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, is_regression ? "MSE" : "CE")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)

        if is_regression
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_skip,
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_skip, total_steps=global_step,
                best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        end
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                         # "val_loss" => val_losses[end], "val_acc" => val_accs[end], "global_step" => global_step)

                         "val_loss" => val_losses[end], "val_acc" => val_accs[end], "global_step" => global_step,
                         "best_val_loss" => best_val_loss,  # per-epoch so hyperband early_terminate can see it
                         "best_val_acc" => best_val_acc)    # sweep metric (classification): matches best/ selection
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end
end


# best-model test eval
opt = nothing; GC.gc(true); CUDA.reclaim()  # free optimizer state
best_cpu = load_best_cpu(ft_model, save_dir)
best_preds, best_trues = if isnothing(best_cpu)
    println("no best/ checkpoint found, best metrics = final model")
    all_preds, all_trues
else
    best_model = fix_gpu_dropout(cu(best_cpu))
    Flux.testmode!(best_model)
    # _, bp, bt = run_test(best_model)

    _, bp, bt = run_test(best_model; which=:best)
    bp, bt
end
# best_metrics = test_metrics(best_preds, best_trues, is_regression)
# final_metrics = test_metrics(all_preds, all_trues, is_regression)
# isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues)

# streaming classification: test = pretrain-unseen cells, *_full = all test cells (as sc_finetune!)
no_agg = (well_acc = NaN, n_wells = 0, wellcl_acc = NaN, n_wellcl = 0)
final_agg = get(test_ga, :final, no_agg)
best_agg = get(test_ga, :best, final_agg)
all_preds_full, all_trues_full, best_preds_full, best_trues_full = all_preds, all_trues, best_preds, best_trues
best_agg_full = best_agg
best_ids = hasproperty(best_agg, :pred_shard) ?
    (shard = basename.(test_paths)[best_agg.pred_shard], cell = best_agg.pred_cell, pt_unseen = best_agg.unseen_mask) : nothing
_u(a) = hasproperty(a, :unseen_mask) ? (well_acc = a.well_acc_u, n_wells = a.n_wells_u, wellcl_acc = a.wellcl_acc_u, n_wellcl = a.n_wellcl_u) : a
has_unseen = is_streaming && !is_regression && !isnothing(unseen) && hasproperty(final_agg, :unseen_mask)
if has_unseen
    all_preds, all_trues = all_preds[final_agg.unseen_mask], all_trues[final_agg.unseen_mask]
    best_preds, best_trues = best_preds[best_agg.unseen_mask], best_trues[best_agg.unseen_mask]
    final_agg, best_agg = _u(final_agg), _u(best_agg)
end
split_tag = !(is_streaming && !is_regression) ? "lvl3" : has_unseen ? "sc_ptunseen" : "sc_full"
best_metrics = test_metrics(best_preds, best_trues, is_regression)
final_metrics = test_metrics(all_preds, all_trues, is_regression)
isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues,
    all_preds_full=best_preds_full, all_trues_full=best_trues_full, split_tag=split_tag,
    test_shard=(isnothing(best_ids) ? String[] : best_ids.shard), test_cell=(isnothing(best_ids) ? Int32[] : best_ids.cell),
    test_pt_unseen=(isnothing(best_ids) ? Bool[] : collect(best_ids.pt_unseen)),
    label_names=((hasproperty(d, :label_to_id) && d.label_to_id isa AbstractDict) ? sort(collect(keys(d.label_to_id)), by=k -> d.label_to_id[k]) : String[]))
println("test (best model, epoch $best_epoch): ", best_metrics)
println("test (final model):         ", final_metrics)

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_val_acc"] = best_val_acc
    wb.summary["best_epoch"] = best_epoch
    for (k, v) in pairs(best_metrics); wb.summary["best_$(k)"] = v; end
    for (k, v) in pairs(final_metrics); wb.summary["final_$(k)"] = v; end
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

log_model(ft_model, save_dir)
log_info(; save_dir=save_dir, train_indices=d.train_idx, val_indices=d.val_idx, test_indices=d.test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=test_input)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

if is_regression
    r2 = 1.0 - sum((all_preds .- all_trues) .^ 2) / sum((all_trues .- mean(all_trues)) .^ 2)
    pearson = cor(all_preds, all_trues)
    rmse = sqrt(mean((all_preds .- all_trues) .^ 2))
    println("R² = $(round(r2, digits=4)), Pearson r = $(round(pearson, digits=4)), RMSE = $(round(rmse, digits=4))")
    if !isnothing(id_baseline)
        println("Identity baseline: R²=$(round(id_baseline.r2, digits=4)), Pearson=$(round(id_baseline.pearson, digits=4)), RMSE=$(round(id_baseline.rmse, digits=4))")
    end
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=finetune_skip, r2=r2, pearson=pearson, rmse=rmse,
               best_r2=best_metrics.r2, best_pearson=best_metrics.pearson, best_rmse=best_metrics.rmse,
               final_r2=final_metrics.r2, final_pearson=final_metrics.pearson, final_rmse=final_metrics.rmse,
               id_r2=isnothing(id_baseline) ? NaN : id_baseline.r2,
               id_pearson=isnothing(id_baseline) ? NaN : id_baseline.pearson,
               id_rmse=isnothing(id_baseline) ? NaN : id_baseline.rmse,
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               # skip=finetune_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               # best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)

               skip=finetune_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               best_well_acc=best_agg.well_acc, best_wellcl_acc=best_agg.wellcl_acc,
               split_tag=split_tag, best_accuracy_fullgroup=(isempty(best_preds_full) ? NaN : mean(best_preds_full .== best_trues_full)),
               best_wellcl_acc_fullgroup=best_agg_full.wellcl_acc,
               final_well_acc=final_agg.well_acc, final_wellcl_acc=final_agg.wellcl_acc,
               best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
end

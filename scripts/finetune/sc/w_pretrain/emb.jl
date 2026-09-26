using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, Extract, FTModels
using LoadSC, ProcessSC

args = load_sc_finetune_args()
config = load_config(args["config"], args)
config["data_format"] = "tahoe_sc"
resolve_model_dir!(config)
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
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))

# SC data loading
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])

# gene selection based on modeltype
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
# load PB data for per-cell SC lvl3 (PCA targets from PB compound-means)
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
is_streaming = d.train_shard_map !== nothing  # false for lvl3 (pseudo-bulked, small)

id_baseline = is_regression ? d.id_baseline : nothing  # identity baseline from the lvl3 loader

# X_val, X_test = d.X_val, d.X_test  # no longer materialized in streaming mode

if config["modeltype"] == "rtf"
    # RTF: data is Int32 gene IDs, top_k truncated
    seq_len = top_k
    n_genes_for_model = d.n_genes  # n_coding — full vocab for embedding lookup
else
    # ETF: data is Float32 expression, HVG already applied in load
    # seq_len = size(X_val, 1)  # no longer materialized in streaming mode
    seq_len = d.n_genes  # n_hvg or n_coding
    n_genes_for_model = n_coding  # pretrained vocab size for weight loading
end

if is_streaming
    # streaming embedding extraction: build pretrained encoder, extract embeddings shard-by-shard
    # then train MLP on the embedding matrix (fits in RAM ~10GB for 256d × 10M cells)

    # 1) build pretrained encoder (same logic as build_embm but without running on X_train)
    ac = FTModels._load_arch_config(config["model_dir"], config)
    state = JLD2.load("$(config["model_dir"])/model_state.jld2")["model_state"]

    # select embedding extraction function based on modeltype/task
    if config["modeltype"] == "etf"
        embed_fn = get_embeds_exp
    elseif config["task"] in ("lrecon", "erecon")
        embed_fn = get_embeds_lrecon
    else
        embed_fn = get_embeds
    end

    # build and load pretrained model
    if config["modeltype"] == "etf" && config["task"] == "lrecon"
        pt_model = ExpLReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    elseif config["modeltype"] == "etf" && config["task"] == "erecon"
        pt_model = ExpEReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    elseif config["modeltype"] == "etf" && config["task"] == "mlm"
        pt_model = ExpModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes_for_model,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"], seq_len=seq_len)
    elseif config["modeltype"] == "rtf" && config["task"] == "lrecon"
        pt_model = RankLReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    elseif config["modeltype"] == "rtf" && config["task"] == "erecon"
        pt_model = RankEReconModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"], seq_len=seq_len)
    else  # rtf + mlm
        pt_model = RankModel(n_genes=n_genes_for_model, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes_for_model,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"], seq_len=seq_len)
    end
    Flux.loadmodel!(pt_model, FTModels._clean_state(state, pt_model))
    pt_model = fix_gpu_dropout(cu(pt_model))
    Flux.testmode!(pt_model)


    embed_dim = ac["embed_dim"]

    # helper: extract embedding from a batch through pretrained encoder
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
            # if task in ("lrecon", "erecon")
            #     embedded = pt_model.embedding(x_gpu)
            #     pos_ids = cu(Int32.(1:size(embedded, 2)))
            #     encoded = embedded .+ pt_model.pos_emb(pos_ids)
            #     dropped = pt_model.emb_dropout(encoded)
            #     transformed = pt_model.transformer(dropped)
            #     emb = cpu(dropdims(mean(transformed, dims=2), dims=2))
            # else
            #     emb = cpu(dropdims(mean(encode(pt_model, x_gpu), dims=2), dims=2))
            # end
            # all rank objectives share the PAD-aware encoder; mean over real tokens only (Models.encode_rank)
            emb = cpu(masked_mean_pool(encode_rank(pt_model, x_gpu)...))
            CUDA.unsafe_free!(x_gpu)
        end
        return emb
    end

    # helper: stream-extract embeddings from shard maps into pre-allocated matrix
    function _stream_extract_embeddings!(output_mat, output_labels, shard_paths, shard_map,
                                          pt_model, n_total, label; n_cls=d.n_classifications)
        col_offset = 0
        for (si, shard_path) in enumerate(shard_paths)
            cell_indices, cell_labels = shard_map[shard_path]
            batches = finetune_batches_from_shard(shard_path, cell_indices, cell_labels,
                                                   token_to_idx, n_coding, top_k,
                                                   config["batch_size"], config["modeltype"], n_cls;
                                                   hvg_idx=hvg_idx, use_oversmpl=false,
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

    # 2) stream-extract train embeddings
    n_train = d.n_train_cells
    train_input = zeros(Float32, embed_dim, n_train)
    y_train = zeros(Float32, d.n_classifications, n_train)
    println("streaming train embedding extraction: $(n_train) cells into ($(embed_dim), $(n_train)) matrix...")
    col_train = _stream_extract_embeddings!(train_input, y_train, d.train_shard_paths, d.train_shard_map,
                                             pt_model, n_train, "train")
    if col_train < n_train
        train_input = train_input[:, 1:col_train]
        y_train = y_train[:, 1:col_train]
    end

    # 3) stream-extract val embeddings
    n_val = d.n_val_cells
    val_input = zeros(Float32, embed_dim, n_val)
    y_val = zeros(Float32, d.n_classifications, n_val)
    println("streaming val embedding extraction: $(n_val) cells...")
    col_val = _stream_extract_embeddings!(val_input, y_val, d.val_shard_paths, d.val_shard_map,
                                           pt_model, n_val, "val")
    if col_val < n_val
        val_input = val_input[:, 1:col_val]
        y_val = y_val[:, 1:col_val]
    end

    # 4) stream-extract test embeddings
    n_test = d.n_test_cells
    test_input = zeros(Float32, embed_dim, n_test)
    y_test = zeros(Float32, d.n_classifications, n_test)
    println("streaming test embedding extraction: $(n_test) cells...")
    col_test = _stream_extract_embeddings!(test_input, y_test, d.test_shard_paths, d.test_shard_map,
                                            pt_model, n_test, "test")
    if col_test < n_test
        test_input = test_input[:, 1:col_test]
        y_test = y_test[:, 1:col_test]
    end

    # free pretrained model from GPU
    pt_model = nothing
    GC.gc(true)
    CUDA.reclaim()

    # 4) build MLP head
    ft_model = Flux.Chain(
        Flux.Dense(embed_dim => config["hidden_dim"], gelu),
        Flux.LayerNorm(config["hidden_dim"]),
        Flux.Dropout(config["drop_prob"]),
        Flux.Dense(config["hidden_dim"] => d.n_classifications))
    ft_model = fix_gpu_dropout(cu(ft_model))
    opt = Flux.setup(Optimisers.AdamW(config["lr"]), ft_model)
else
    # non-streaming path: use build_embm as before (lvl3 pseudo-bulked data)
    X_train = d.X_train
    X_val, X_test = d.X_val, d.X_test
    ft_model, train_input, val_input, test_input = build_embm(config, X_train, X_test,
                                                    n_genes_for_model, d.n_classifications; X_val=X_val,
                                                    seq_len=seq_len)
    y_train = d.y_train
    y_val = d.y_val
    y_test = d.y_test
    opt = Flux.setup(Optimisers.AdamW(config["lr"]), ft_model)
end

# save dir
dataset_tag = joinpath("tahoe", "sc")
save_dir = joinpath("results", dataset_tag, "finetune", "w_pretrain", config["level"],
                    config["modeltype"], config["task"], "emb", timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
wandb = init_wandb(config, "SC-FT-Aug", "emb_sc_$(config["modeltype"])_$(config["level"])$(seed_tag)_$(timestamp)")
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
best_val_loss = Inf32
best_epoch = 0
# build oversampling indices from y_train if needed (for streaming, d.cidx_dict not available)
if use_oversmpl
    if is_streaming
        # build class→indices dict from y_train one-hot matrix
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

n_total_epochs = if use_max_steps
    bpe = div(size(train_input, 2), config["batch_size"])
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

# test-set eval for model `m` (used for the final model at the last epoch and for the reloaded best model)
function run_test(m)
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
    num_batches = div(n_train, config["batch_size"])
    perm = use_oversmpl ? nothing : randperm(n_train)

    for i in 1:num_batches
        if use_oversmpl
            batch_idx = [rand(cidx_dict[rand(cs)]) for _ in 1:config["batch_size"]]
        else
            s = (i - 1) * config["batch_size"] + 1
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

    # val eval (every epoch for checkpt selection)
    Flux.testmode!(ft_model)
    val_eval_losses = Float32[]
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
        end
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        final_test_loss, epoch_preds, epoch_trues = run_test(ft_model)
        push!(test_losses, final_test_loss)
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
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
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_skip, total_steps=global_step,
                best_epoch=best_epoch, best_val_loss=best_val_loss)
        end
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                         "val_loss" => val_losses[end], "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end
end


# best-model test eval: reload the best-val checkpoint (best/) and re-run the test set
# all_preds / all_trues above come from the final model
opt = nothing; GC.gc(true); CUDA.reclaim()   # free optimizer state before loading a second model copy
best_cpu = load_best_cpu(ft_model, save_dir)
best_preds, best_trues = if isnothing(best_cpu)
    println("no best/ checkpoint found, best metrics = final model")
    all_preds, all_trues
else
    best_model = fix_gpu_dropout(cu(best_cpu))
    Flux.testmode!(best_model)
    _, bp, bt = run_test(best_model)
    bp, bt
end
best_metrics = test_metrics(best_preds, best_trues, is_regression)
final_metrics = test_metrics(all_preds, all_trues, is_regression)
isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues)
println("test (best model, epoch $best_epoch): ", best_metrics)
println("test (final model):         ", final_metrics)

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
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
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=finetune_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               best_epoch=best_epoch, best_val_loss=best_val_loss)
end

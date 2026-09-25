using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, FTModels
using LoadSC, ProcessSC

args = load_sc_finetune_args()
config = load_config(args["config"], args,
                     hp_section=["finetune", "no_pretrain", args["modeltype"], args["level"]],
                     dataset="tahoe_sc")
config["data_format"] = "tahoe_sc"
resolve_lvl3_cells!(config)
# resolve_model_dir!(config)  # no pretrain weights needed

# seed
seed = get(config, "seed", nothing)
if !isnothing(seed)
    Random.seed!(seed)
    CUDA.seed!(seed)
    println("Random seed: $seed")
end

# is_regression = false  # SC finetune: lvl1/lvl2 classification only
is_regression = config["level"] == "lvl3"
use_oversmpl = config["level"] == "lvl2" && !is_regression

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

start_time = now()
# timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))

# data — SC shard loading (RTF mode returns Int32 gene IDs in rank order)
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])

top_k = get(config, "top_k", 1024)
# d = load_sc_finetune_data(all_shards, config["level"], token_to_idx, n_coding, top_k, "rtf";
#                            pb_data_path=get(config, "pb_data_path", ""),
#                            subset_shards=get(config, "subset_shards", 0),
#                            process_cell_topk_flat_fn=process_cell_topk_flat,
#                            cell_to_dense_flat_fn=cell_to_dense_flat!,
#                            oversmpl_fn=oversmpl,
#                            source_cell=get(config, "source_cell", ""),
#                            target_cell=get(config, "target_cell", ""),
#                            dose=get(config, "dose", ""),
#                            meta_dir=get(config, "meta_dir", ""),
#                            regression_pairs_fn=get_regression_pairs_pca)
# load PB data for per-cell SC lvl3 (PCA targets from PB compound-means)
# sc_lvl3_percell = get(config, "sc_lvl3_percell", false)
sc_lvl3_percell = !get(config, "sc_lvl3_pseudobulk", false)  # per-cell is the default
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

d = load_sc_finetune_data_streaming(all_shards, config["level"], token_to_idx, n_coding, top_k, "rtf";
                           pb_data_path=get(config, "pb_data_path", ""),
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
                           actual_modeltype=config["modeltype"],
                           identity_baseline_fn=identity_baseline)
is_streaming = d.train_shard_map !== nothing  # false for lvl3 (pseudo-bulked, small)

# cap val/test eval to first ft_eval_shards shards (0 = all), same subset for every run
# n_eval = something(get(config, "n_eval_shards", 0), 0)  # inherited default.toml pretrain value (10)
n_eval = something(get(config, "ft_eval_shards", 0), 0)
if is_streaming
    val_paths  = n_eval > 0 ? d.val_shard_paths[1:min(n_eval, length(d.val_shard_paths))]   : d.val_shard_paths
    test_paths = n_eval > 0 ? d.test_shard_paths[1:min(n_eval, length(d.test_shard_paths))] : d.test_shard_paths
    println("eval shards: val=$(length(val_paths))/$(length(d.val_shard_paths)) test=$(length(test_paths))/$(length(d.test_shard_paths))")
end

id_baseline = is_regression ? d.id_baseline : nothing  # identity baseline from the lvl3 loader

# convert RTF gene-id tokens to inverse ranks (position g = rank of gene g)
# X_rtf is (top_k, n_samples) Int32 matrix where X_rtf[rank, sample] = gene_id
# inv is (n_coding, n_samples) Float32 matrix where inv[gene_id, sample] = rank
function sc_inverse_ranks(X_rtf::Matrix{Int32}, n_coding::Int)
    inv = zeros(Float32, n_coding, size(X_rtf, 2))
    for j in axes(X_rtf, 2)
        for r in axes(X_rtf, 1)
            g = X_rtf[r, j]
            if g > 0 && g <= n_coding
                inv[g, j] = Float32(r)
            end
        end
    end
    return inv
end

# for streaming: all inverse ranks computed per-batch via sc_inverse_ranks_batch()
# for non-streaming (lvl3): compute upfront on materialized data
if is_streaming
    println("streaming mode: all inverse ranks (train/val/test) computed per-batch")
    # X_val   = sc_inverse_ranks(d.X_val, n_coding)   ./ Float32(n_coding)  # removed: val/test no longer materialized
    # X_test  = sc_inverse_ranks(d.X_test, n_coding)  ./ Float32(n_coding)
    X_train = nothing  # not materialized
elseif eltype(d.X_train) == Float32
    # lvl3 percell/pseudo-bulk: data is already Float32 ranks, just normalize
    println("lvl3 mode: data already ranked, normalizing by n_coding=$n_coding")
    X_train = d.X_train ./ Float32(n_coding)
    X_val   = d.X_val   ./ Float32(n_coding)
    X_test  = d.X_test  ./ Float32(n_coding)
else
    println("converting RTF tokens to inverse ranks (n_coding=$n_coding)...")
    X_train = sc_inverse_ranks(d.X_train, n_coding) ./ Float32(n_coding)
    X_val   = sc_inverse_ranks(d.X_val, n_coding)   ./ Float32(n_coding)
    X_test  = sc_inverse_ranks(d.X_test, n_coding)  ./ Float32(n_coding)
end
n_genes = n_coding  # MLP input dim = full gene space
n_classifications = d.n_classifications
# y_val, y_test = d.y_val, d.y_test  # no longer materialized in streaming mode
if !is_streaming
    y_val = d.y_val
    y_test = d.y_test
    y_train = d.y_train
end
train_idx, val_idx, test_idx = d.train_idx, d.val_idx, d.test_idx
# cidx_dict, cs = d.cidx_dict, d.cs  # oversampling handled per-shard in streaming mode
println("inverse ranks done: input dim = $n_genes")

# model — MLP with linearly interpolated layer sizes
if config["modeltype"] == "rlog"
    # single linear layer, no activation, no dropout: logistic reg (lvl1/2, CE) / linear reg (lvl3, MSE)
    config["lr"] = 0.001
    model = Flux.Chain(Flux.Dense(n_genes => n_classifications))
    model = cu(model)
else
    # nonlinear MLP: tapered layers with relu + dropout
    sizes = [round(Int, n_genes + (n_classifications - n_genes) * i / (config["n_layers"] + 1))
             for i in 0:config["n_layers"]+1]
    layers = []
    for i in 1:length(sizes)-1
        push!(layers, Flux.Dense(sizes[i] => sizes[i+1], i < length(sizes)-1 ? relu : identity))
        if i < length(sizes) - 1
            push!(layers, Flux.Dropout(config["drop_prob"]))
        end
    end
    model = Flux.Chain(layers...)
    model = fix_gpu_dropout(cu(model))
end
opt = Flux.setup(Optimisers.AdamW(config["lr"]), model)

# save dir
dataset_tag = joinpath("tahoe", "sc")
# save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], "rmlp", timestamp)
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
wandb = init_wandb(config, "SC-FT-Aug", "$(config["modeltype"])_nopt_sc_$(config["level"])$(seed_tag)_$(timestamp)")
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
n_total_epochs = if use_max_steps
    if is_streaming
        bpe = cld(d.n_train_cells, config["batch_size"])
    else
        bpe = div(size(X_train, 2), config["batch_size"])
    end
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)

    # train epoch
    Flux.trainmode!(model)
    epoch_losses = Float32[]

    if is_streaming
        # shard-level streaming training with per-batch inverse rank conversion
        shuffled_shards = shuffle(d.train_shard_paths)
        for (si, shard_path) in enumerate(shuffled_shards)
            done && break
            cell_indices, cell_labels = d.train_shard_map[shard_path]
            batches = finetune_batches_from_shard(shard_path, cell_indices, cell_labels,
                                                   token_to_idx, n_coding, top_k,
                                                   config["batch_size"], "rtf", n_classifications;
                                                   hvg_idx=nothing, use_oversmpl=use_oversmpl,
                                                   process_cell_topk_flat_fn=process_cell_topk_flat,
                                                   cell_to_dense_flat_fn=cell_to_dense_flat!)
            for (x_batch, y_batch) in batches
                # convert RTF gene IDs to inverse ranks per batch
                x_inv = sc_inverse_ranks_batch(x_batch, n_coding)
                x_gpu = CuArray(x_inv)
                y_gpu = CuArray(y_batch)

                lv, grads = Flux.withgradient(model) do m
                    preds = m(x_gpu)
                    is_regression ? Flux.mse(preds, y_gpu) : Flux.logitcrossentropy(preds, y_gpu)
                end
                Flux.update!(opt, model, grads[1])
                CUDA.unsafe_free!(x_gpu)
                CUDA.unsafe_free!(y_gpu)
                push!(epoch_losses, Float32(cpu(lv)))
                global global_step += 1
                if use_max_steps && global_step >= ft_step_limit
                    global done = true; break
                end
            end
            if si % 100 == 0
                println("  epoch $epoch shard $si/$(length(shuffled_shards)) step=$global_step")
                flush(stdout)
            end
        end
    else
        # non-streaming path (lvl3 pseudo-bulked data)
        n_train = size(X_train, 2)
        num_batches = div(n_train, config["batch_size"])
        perm = randperm(n_train)

        for i in 1:num_batches
            s = (i - 1) * config["batch_size"] + 1
            e = min(s + config["batch_size"] - 1, n_train)
            batch_idx = perm[s:e]

            x_gpu = cu(X_train[:, batch_idx])
            y_gpu = cu(d.y_train[:, batch_idx])

            lv, grads = Flux.withgradient(model) do m
                preds = m(x_gpu)
                is_regression ? Flux.mse(preds, y_gpu) : Flux.logitcrossentropy(preds, y_gpu)
            end
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, Float32(cpu(lv)))
            global global_step += 1
            if use_max_steps && global_step >= ft_step_limit
                global done = true; break
            end
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval (every epoch for checkpt selection)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    if is_streaming
        # println("  epoch $epoch: val eval ($(length(d.val_shard_paths)) shards)"); flush(stdout)
        println("  epoch $epoch: val eval ($(length(val_paths)) shards)"); flush(stdout)
        # for (vi, shard_path) in enumerate(d.val_shard_paths)
        for (vi, shard_path) in enumerate(val_paths)
            cell_indices, cell_labels = d.val_shard_map[shard_path]
            batches = finetune_batches_from_shard(shard_path, cell_indices, cell_labels,
                                                   token_to_idx, n_coding, top_k,
                                                   config["batch_size"], "rtf", d.n_classifications;
                                                   hvg_idx=nothing, use_oversmpl=false,
                                                   process_cell_topk_flat_fn=process_cell_topk_flat,
                                                   cell_to_dense_flat_fn=cell_to_dense_flat!)
            for (x_batch, y_batch) in batches
                x_inv = sc_inverse_ranks_batch(x_batch, n_coding)
                x_gpu = CuArray(x_inv)
                y_gpu = CuArray(y_batch)
                logits = model(x_gpu)
                if is_regression
                    push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
                else
                    push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
                end
                CUDA.unsafe_free!(x_gpu); CUDA.unsafe_free!(y_gpu)
            end
            # if vi % 200 == 0; println("    val shard $vi/$(length(d.val_shard_paths))"); flush(stdout); end
            if vi % 200 == 0; println("    val shard $vi/$(length(val_paths))"); flush(stdout); end
        end
    else
        n_val = size(X_val, 2)
        for s in 1:config["batch_size"]:n_val
            e = min(s + config["batch_size"] - 1, n_val)
            x_gpu = cu(X_val[:, s:e])
            y_gpu = cu(y_val[:, s:e])
            logits = model(x_gpu)
            if is_regression
                push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
            else
                push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            end
        end
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        eval_losses = Float32[]
        if is_streaming
            # println("  test eval ($(length(d.test_shard_paths)) shards)"); flush(stdout)
            println("  test eval ($(length(test_paths)) shards)"); flush(stdout)
            # for (ti, shard_path) in enumerate(d.test_shard_paths)
            for (ti, shard_path) in enumerate(test_paths)
                cell_indices, cell_labels = d.test_shard_map[shard_path]
                batches = finetune_batches_from_shard(shard_path, cell_indices, cell_labels,
                                                       token_to_idx, n_coding, top_k,
                                                       config["batch_size"], "rtf", d.n_classifications;
                                                       hvg_idx=nothing, use_oversmpl=false,
                                                       process_cell_topk_flat_fn=process_cell_topk_flat,
                                                       cell_to_dense_flat_fn=cell_to_dense_flat!)
                for (x_batch, y_batch) in batches
                    x_inv = sc_inverse_ranks_batch(x_batch, n_coding)
                    x_gpu = CuArray(x_inv)
                    y_gpu = CuArray(y_batch)
                    logits = model(x_gpu)
                    if is_regression
                        push!(eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
                        append!(epoch_preds, vec(cpu(logits)))
                        append!(epoch_trues, vec(cpu(y_gpu)))
                    else
                        push!(eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
                        append!(epoch_preds, Flux.onecold(cpu(logits)))
                        append!(epoch_trues, Flux.onecold(y_batch))
                    end
                    CUDA.unsafe_free!(x_gpu); CUDA.unsafe_free!(y_gpu)
                end
                # if ti % 200 == 0; println("    test shard $ti/$(length(d.test_shard_paths))"); flush(stdout); end
                if ti % 200 == 0; println("    test shard $ti/$(length(test_paths))"); flush(stdout); end
            end
        else
            n_test = size(X_test, 2)
            for s in 1:config["batch_size"]:n_test
                e = min(s + config["batch_size"] - 1, n_test)
                x_gpu = cu(X_test[:, s:e])
                y_gpu = cu(y_test[:, s:e])
                logits = model(x_gpu)
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
        end
        push!(test_losses, mean(eval_losses))
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        best_dir = joinpath(save_dir, "best")
        mkpath(best_dir)
        log_model(model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, is_regression ? "MSE" : "CE")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)
        if is_regression
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=mlp_skip,
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=mlp_skip, total_steps=global_step,
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

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_epoch"] = best_epoch
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

log_model(model, save_dir)
log_info(; save_dir=save_dir, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=nothing)  # streaming: X_test not materialized

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
               skip=mlp_skip, r2=r2, pearson=pearson, rmse=rmse,
               id_r2=isnothing(id_baseline) ? NaN : id_baseline.r2,
               id_pearson=isnothing(id_baseline) ? NaN : id_baseline.pearson,
               id_rmse=isnothing(id_baseline) ? NaN : id_baseline.rmse,
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=mlp_skip, accuracy=acc, total_steps=global_step,
               best_epoch=best_epoch, best_val_loss=best_val_loss)
end

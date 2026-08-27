using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics, Functors
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
using Preprocess, Models, Train, Log, Plot, Args, Config, ProcessLabels

args = load_pretrain_args()
config = load_config(args["config"], args)

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

use_exp = config["modeltype"] == "etf"

start_time = now()

function get_target_embeds(model, x_clean, use_exp)
    if use_exp
        projected = model.proj(reshape(x_clean, 1, size(x_clean)...))
        gene_ids = cu(Int32.(1:size(x_clean, 1)))
        encoded = projected .+ model.pos_emb(gene_ids)
    else
        embedded = model.embedding(x_clean)
        pos_ids = cu(Int32.(1:size(embedded, 2)))
        encoded = embedded .+ model.pos_emb(pos_ids)
    end
    return model.transformer(model.emb_dropout(encoded))
end

fmt = get(config, "data_format", "tahoe")
data_key = fmt == "lincs" ? "filtered_data" : "df"
data = load(config["data_path"])[data_key]
if fmt == "lincs"
    data_expr = data.expr
    meta_df = data.inst
else  # tahoe
    data_expr = reduce(hcat, data.expr)
    meta_df = data
end


if config["subset_ratio"] < 1.0
    n_total = size(data_expr, 2)
    data_expr, subset_idx = downsmpl(data_expr, meta_df, config["subset_ratio"], fmt)
    println("subset: $(length(subset_idx))/$(n_total) samples")
end

n_hvg = get(config, "n_hvg", 0)
if n_hvg > 0 && n_hvg < size(data_expr, 1)
    n_orig = size(data_expr, 1)
    data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
    println("HVG filter: $(n_orig) → $(n_hvg) genes")
end

gene_medians = vec(median(data_expr, dims=2)) .+ 1f-10
X_ranks = rank_genes(data_expr, gene_medians)

n_genes = size(X_ranks, 1)
MASK_ID = n_genes + 1

# splits
# _, _, train_indices, test_indices = ttsplit(X_ranks, 0.2f0)
_, _, _, train_indices, val_indices, test_indices = tvsplit(X_ranks, 0.1f0, 0.1f0)

if use_exp
    X_expr = Float32.(data_expr)
    X_train = X_expr[:, train_indices]
    X_val = X_expr[:, val_indices]
    X_test = X_expr[:, test_indices]
    X_ranks_train = X_ranks[:, train_indices]
    X_ranks_val = X_ranks[:, val_indices]
    X_ranks_test = X_ranks[:, test_indices]
else
    X_train = X_ranks[:, train_indices]
    X_val = X_ranks[:, val_indices]
    X_test = X_ranks[:, test_indices]
end
X_ranks_val = X_ranks[:, val_indices]
X_ranks_test = X_ranks[:, test_indices]
if use_exp
    inv_ranks_val = inverse_ranks(X_ranks_val)
    inv_ranks_test = inverse_ranks(X_ranks_test)
end

# model
model = if use_exp
    ExpLReconModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                   n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                   dropout_prob=config["drop_prob"])
else
    RankLReconModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                    dropout_prob=config["drop_prob"])
end
model = cu(model)
model = fix_gpu_dropout(model)

ema_model = deepcopy(model)
Flux.testmode!(ema_model)

# opt = Flux.setup(Adam(config["lr"]), model)
# opt = Flux.setup(AdamW(config["lr"]), model)
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)

# mask val
X_val_masked = similar(X_val)
if use_exp
    val_corrupt_mask = falses(size(X_val))
    corrupt_expr!(X_val_masked, val_corrupt_mask, X_val, config["mask_ratio"])
else
    y_val_masked = similar(X_val)
    mask_input!(X_val_masked, y_val_masked, X_val, config["mask_ratio"], -100, MASK_ID, false)
end

# mask test
X_test_masked = similar(X_test)
if use_exp
    test_corrupt_mask = falses(size(X_test))
    corrupt_expr!(X_test_masked, test_corrupt_mask, X_test, config["mask_ratio"])
else
    y_test_masked = similar(X_test)
    mask_input!(X_test_masked, y_test_masked, X_test, config["mask_ratio"], -100, MASK_ID, false)
end

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
fmt = get(config, "data_format", "tahoe")
dataset_tag = fmt == "lincs" ? joinpath("lincs") : joinpath("tahoe", "pb")
save_dir = joinpath("results", dataset_tag, "pretrain", "lrecon", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")
if n_hvg > 0
    jldsave(joinpath(save_dir, "hvg_indices.jld2"); hvg_idx=hvg_idx)
end

wandb = init_wandb(config, "PB-PT-Aug", "lrecon_$(fmt)_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

# train
train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
target_variances = Float32[]
gene_error_sums = zeros(Float32, n_genes)
gene_error_counts = zeros(Int, n_genes)
rank_error_sums = zeros(Float32, n_genes)
rank_error_counts = zeros(Int, n_genes)

global_step = 0
use_max_steps = config["max_steps"] > 0
done = false
# best_test_loss = Inf32
best_val_loss = Inf32
best_epoch = 0

n_total_epochs = if use_max_steps
    bpe = cld(size(X_train, 2), config["batch_size"])
    cld(config["max_steps"], bpe)
else
    config["n_epochs"]
end
warmup_epochs = max(1, div(n_total_epochs, 10))
X_train_masked = similar(X_train)
if use_exp
    train_corrupt_mask = falses(size(X_train))
else
    y_train_masked = similar(X_train)
end
saved_preds = Vector{Float32}[]
saved_targets = Vector{Float32}[]
saved_positions = Int32[]

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)
    Optimisers.adjust!(opt, compute_lr(epoch, n_total_epochs, config["lr"], warmup_epochs))

    if use_exp
        corrupt_expr!(X_train_masked, train_corrupt_mask, X_train, config["mask_ratio"])
    else
        mask_input!(X_train_masked, y_train_masked, X_train, config["mask_ratio"], -100, MASK_ID, false)
    end

    # train epoch
    Flux.trainmode!(model)
    epoch_losses = Float32[]

    for start_idx in 1:config["batch_size"]:size(X_train_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_train_masked, 2))

        x_batch = CuArray(X_train_masked[:, start_idx:end_idx])
        x_clean_batch = CuArray(X_train[:, start_idx:end_idx])
        target_embeds = get_target_embeds(ema_model, x_clean_batch, use_exp)

        if use_exp
            m_batch = CuArray(Float32.(train_corrupt_mask[:, start_idx:end_idx]))
            mask_2d = m_batch
        else
            y_batch = CuArray(y_train_masked[:, start_idx:end_idx])
            mask_2d = Float32.(y_batch .!= -100)
        end

        l_val, grads = Flux.withgradient(model) do m
            masked_lrecon_loss(m, x_batch, target_embeds, mask_2d)[1]
        end
        Flux.update!(opt, model, grads[1])
        ema_update!(ema_model, model, Float32(config["ema_decay"]))
        push!(epoch_losses, l_val)
        global global_step += 1
        if use_max_steps && global_step >= config["max_steps"]
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # collapse detect
    sample_clean = CuArray(X_train[:, 1:min(config["batch_size"], size(X_train, 2))])
    sample_targets = get_target_embeds(ema_model, sample_clean, use_exp)
    tgt_var = mean(cpu(var(sample_targets, dims=3)))
    push!(target_variances, Float32(tgt_var))
    if tgt_var < 1f-6
        println("UH OH epoch $epoch: target embedding variance = $tgt_var")
    end

    # val eval (every epoch — used for checkpointing and sweep selection)
    val_eval_losses = Float32[]
    for start_idx in 1:config["batch_size"]:size(X_val_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_val_masked, 2))
        x_batch = CuArray(X_val_masked[:, start_idx:end_idx])
        x_clean_batch = CuArray(X_val[:, start_idx:end_idx])
        target_embeds = get_target_embeds(ema_model, x_clean_batch, use_exp)
        if use_exp
            m_batch = CuArray(Float32.(val_corrupt_mask[:, start_idx:end_idx]))
            mask_2d = m_batch
        else
            y_batch = CuArray(y_val_masked[:, start_idx:end_idx])
            mask_2d = Float32.(y_batch .!= -100)
        end
        loss_val, _, _ = masked_lrecon_loss(ema_model, x_batch, target_embeds, mask_2d)
        push!(val_eval_losses, cpu(loss_val))
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only — held out for reporting)
    is_last = is_last || done
    eval_losses = Float32[]

    if is_last
        for start_idx in 1:config["batch_size"]:size(X_test_masked, 2)
            end_idx = min(start_idx + config["batch_size"] - 1, size(X_test_masked, 2))

            x_batch = CuArray(X_test_masked[:, start_idx:end_idx])
            x_clean_batch = CuArray(X_test[:, start_idx:end_idx])
            target_embeds = get_target_embeds(ema_model, x_clean_batch, use_exp)

            if use_exp
                m_batch = CuArray(Float32.(test_corrupt_mask[:, start_idx:end_idx]))
                mask_2d = m_batch
                mask_cpu = test_corrupt_mask[:, start_idx:end_idx]
            else
                y_batch = CuArray(y_test_masked[:, start_idx:end_idx])
                mask_bool = y_batch .!= -100
                mask_2d = Float32.(mask_bool)
                mask_cpu = cpu(mask_bool)
            end

            # loss_val, decoded_masked, tgt_masked = masked_lrecon_loss(model, x_batch, target_embeds, mask_2d)
            loss_val, decoded_masked, tgt_masked = masked_lrecon_loss(ema_model, x_batch, target_embeds, mask_2d)
            push!(eval_losses, cpu(loss_val))

            if !isnothing(decoded_masked)
                dec_cpu = cpu(decoded_masked)
                tgt_cpu = cpu(tgt_masked)
                ranks_batch = X_ranks_test[:, start_idx:min(end_idx, size(X_ranks_test, 2))]
                inv_ranks_batch = use_exp ? inv_ranks_test[:, start_idx:min(end_idx, size(inv_ranks_test, 2))] : nothing
                embed_dim_local = size(dec_cpu, 1)
                masked_idx = 0
                for j in 1:size(mask_cpu, 2)
                    for pos in 1:size(mask_cpu, 1)
                        if mask_cpu[pos, j]
                            masked_idx += 1
                            push!(saved_preds, dec_cpu[:, masked_idx])
                            push!(saved_targets, tgt_cpu[:, masked_idx])
                            push!(saved_positions, Int32(pos))
                            emb_mse = sum((dec_cpu[:, masked_idx] .- tgt_cpu[:, masked_idx]) .^ 2) / embed_dim_local
                            if use_exp
                                gene_error_sums[pos] += emb_mse
                                gene_error_counts[pos] += 1
                                r = inv_ranks_batch[pos, j]
                                rank_error_sums[r] += emb_mse
                                rank_error_counts[r] += 1
                            else
                                rank_error_sums[pos] += emb_mse
                                rank_error_counts[pos] += 1
                                gene_id = ranks_batch[pos, j]
                                gene_error_sums[gene_id] += emb_mse
                                gene_error_counts[gene_id] += 1
                            end
                        end
                    end
                end
            end
        end
    end
    if !isempty(eval_losses)
        push!(test_losses, mean(eval_losses))
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                        "val_loss" => val_losses[end],
                        "target_variance" => target_variances[end],
                        "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end

    # if test_losses[end] < best_test_loss
    #     global best_test_loss = test_losses[end]
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(ema_model, joinpath(save_dir, "best"), config)
    end
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "MSE loss")

pred_matrix = reduce(hcat, saved_preds)
target_matrix = reduce(hcat, saved_targets)
jldsave(joinpath(save_dir, "lrecon_diagnostics.jld2");
    preds=pred_matrix, targets=target_matrix, positions=saved_positions,
    target_variances=target_variances)

plot_per_gene_error(gene_error_sums, gene_error_counts, n_genes, save_dir,
                    "mean embedding MSE", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, n_genes, save_dir,
                           "mean embedding MSE", "per_rank_error")

# log_model(ema_model, save_dir)
log_model(ema_model, save_dir, config)
log_info(; save_dir=save_dir, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           target_variances=target_variances,
           X_test_masked=X_test_masked,
           y_test_masked=use_exp ? test_corrupt_mask : y_test_masked,
           X_test=use_exp ? X_expr[:, test_indices] : X_test)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, total_steps=global_step,
           best_epoch=best_epoch, best_val_loss=best_val_loss)

wb !== nothing && wandb.finish()

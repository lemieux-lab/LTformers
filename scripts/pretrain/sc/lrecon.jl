using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics, Functors
using ProgressBars, CairoMakie, StatsBase

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using Preprocess, Models, Train, Log, Plot, Args, Config
using LoadSC, ProcessSC

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

coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])

all_shards = list_shards(config["data_dir"])
train_shards, test_shards = shard_train_test_split(all_shards, 0.2)
println("Shards: $(length(train_shards)) train, $(length(test_shards)) test")

top_k = config["top_k"]
MASK_ID = Int32(n_coding + 1)

model = if use_exp
    ExpLReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                   n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                   dropout_prob=config["drop_prob"])
else
    RankLReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                    dropout_prob=config["drop_prob"])
end
model = cu(model)

ema_model = deepcopy(model)
Flux.testmode!(ema_model)

opt = Flux.setup(Adam(config["lr"]), model)

# mask
X_masked_rtf = Matrix{Int32}(undef, top_k, config["batch_size"])
y_masked_rtf = Matrix{Int32}(undef, top_k, config["batch_size"])
X_masked_etf = Matrix{Float32}(undef, top_k, config["batch_size"])
corrupt_mask_buf = falses(top_k, config["batch_size"])

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("results", "tahoe", "sc", "pretrain", "lrecon", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

wandb = init_wandb(config, "SC-PT-Aug", "lrecon_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

train_losses = Float32[]
test_losses = Float32[]
target_variances = Float32[]
saved_preds = Vector{Float32}[]
saved_targets = Vector{Float32}[]
saved_positions = Vector{Int}[]
gene_error_sums = zeros(Float32, n_coding)
gene_error_counts = zeros(Int, n_coding)
rank_error_sums = zeros(Float32, n_coding)
rank_error_counts = zeros(Int, n_coding)

global_step = 0
use_max_steps = config["max_steps"] > 0
done = false
best_test_loss = Inf32
best_epoch = 0

n_total_epochs = if use_max_steps
    n_sample = min(5, length(train_shards))
    sample_shards = train_shards[1:n_sample]
    local total_cells = 0
    for sp in sample_shards
        shard = load_shard_pyarrow(sp)
        total_cells += shard.n_cells
    end
    avg_cells_per_shard = total_cells / n_sample
    est_cells_per_epoch = avg_cells_per_shard * length(train_shards)
    bpe = cld(Int(round(est_cells_per_epoch)), config["batch_size"])
    cld(config["max_steps"], bpe)
else
    config["n_epochs"]
end
warmup_epochs = max(1, div(n_total_epochs, 10))

collapse_check_batch = nothing

for epoch in 1:n_total_epochs
    done && break
    lr = compute_lr(epoch, n_total_epochs, config["lr"], warmup_epochs)
    Optimisers.adjust!(opt, lr)

    # train
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    shuffled_train = shuffle(train_shards)

    for (si, shard_path) in enumerate(shuffled_train)
        done && break
        shard_modeltype = use_exp ? "etf" : "rtf"
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype=shard_modeltype,
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)

        for batch in batches
            if use_exp
                batch_ids, batch_expr = batch
                bs = size(batch_expr, 2)
                xm = @view X_masked_etf[:, 1:bs]
                cm = @view corrupt_mask_buf[:, 1:bs]
                corrupt_expr!(xm, cm, batch_expr, config["mask_ratio"])
                x_gpu = CuArray(xm)
                x_clean_gpu = CuArray(batch_expr)
                mask_2d = CuArray(Float32.(cm))
            else
                bs = size(batch, 2)
                xm = @view X_masked_rtf[:, 1:bs]
                ym = @view y_masked_rtf[:, 1:bs]
                sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
                x_gpu = CuArray(xm)
                x_clean_gpu = CuArray(batch)
                mask_2d = Float32.(CuArray(ym) .!= -100)
            end

            target_embeds = get_target_embeds(ema_model, x_clean_gpu, use_exp)

            l_val, grads = Flux.withgradient(model) do m
                masked_lrecon_loss(m, x_gpu, target_embeds, mask_2d)[1]
            end
            Flux.update!(opt, model, grads[1])
            ema_update!(ema_model, model, Float32(config["ema_decay"]))
            push!(epoch_losses, l_val)

            # save first clean batch for collapse detection
            if isnothing(collapse_check_batch)
                if use_exp
                    global collapse_check_batch = copy(batch_expr)
                else
                    global collapse_check_batch = copy(batch)
                end
            end

            global global_step += 1
            if use_max_steps && global_step >= config["max_steps"]
                global done = true; break
            end
        end

        if si % 50 == 0
            println("  epoch $epoch shard $si/$(length(shuffled_train)) step=$global_step loss=$(round(mean(epoch_losses[max(1,end-49):end]), digits=4))")
        end
    end
    push!(train_losses, mean(epoch_losses))

    # collapse detection
    if !isnothing(collapse_check_batch)
        sample_clean = CuArray(collapse_check_batch)
        sample_targets = get_target_embeds(ema_model, sample_clean, use_exp)
        tgt_var = mean(cpu(var(sample_targets, dims=3)))
        push!(target_variances, Float32(tgt_var))
        if tgt_var < 1f-6
            println("WARNING epoch $epoch: target embedding variance = $tgt_var — possible collapse")
        end
    end

    # eval
    Flux.testmode!(model)
    eval_losses = Float32[]
    is_last = (epoch == n_total_epochs) || done
    # eval all test shards on last epoch, subset otherwise
    eval_shards = if is_last
        test_shards
    else
        shuffle(test_shards)[1:min(config["n_eval_shards"], length(test_shards))]
    end

    for shard_path in eval_shards
        shard_modeltype = use_exp ? "etf" : "rtf"
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype=shard_modeltype,
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)
        for batch in batches
            if use_exp
                batch_ids, batch_expr = batch
                bs = size(batch_expr, 2)
                xm = Matrix{Float32}(undef, top_k, bs)
                cm = falses(top_k, bs)
                corrupt_expr!(xm, cm, batch_expr, config["mask_ratio"])
                x_gpu = CuArray(xm)
                x_clean_gpu = CuArray(batch_expr)
                mask_2d = CuArray(Float32.(cm))
            else
                bs = size(batch, 2)
                xm = Matrix{Int32}(undef, top_k, bs)
                ym = Matrix{Int32}(undef, top_k, bs)
                sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
                x_gpu = CuArray(xm)
                x_clean_gpu = CuArray(batch)
                mask_2d = Float32.(CuArray(ym) .!= -100)
            end
            target_embeds = get_target_embeds(ema_model, x_clean_gpu, use_exp)
            loss_val, preds_embed, targets_embed = masked_lrecon_loss(model, x_gpu, target_embeds, mask_2d)
            push!(eval_losses, cpu(loss_val))

            if is_last && !isnothing(preds_embed)
                dec_cpu = cpu(preds_embed)
                tgt_cpu = cpu(targets_embed)
                if use_exp
                    mask_cpu_bool = cpu(mask_2d) .> 0f0
                else
                    mask_cpu_bool = cpu(CuArray(ym) .!= -100)
                end
                embed_dim_local = size(dec_cpu, 1)

                masked_idx = 0
                for j in 1:bs
                    for pos in 1:top_k
                        if mask_cpu_bool[pos, j]
                            masked_idx += 1
                            push!(saved_preds, dec_cpu[:, masked_idx])
                            push!(saved_targets, tgt_cpu[:, masked_idx])
                            push!(saved_positions, Int32(pos))
                            emb_mse = sum((dec_cpu[:, masked_idx] .- tgt_cpu[:, masked_idx]) .^ 2) / embed_dim_local
                            gene_error_sums[pos] += emb_mse
                            gene_error_counts[pos] += 1
                            rank_error_sums[pos] += emb_mse
                            rank_error_counts[pos] += 1
                        end
                    end
                end
            end
        end
    end
    push!(test_losses, mean(eval_losses))

    println("epoch $epoch/$n_total_epochs | train=$(round(train_losses[end], digits=4)) test=$(round(test_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")

    if wb !== nothing
        wb.log(Dict("epoch" => epoch, "train_loss" => train_losses[end],
                     "test_loss" => test_losses[end],
                     "target_variance" => isempty(target_variances) ? NaN : target_variances[end],
                     "global_step" => global_step))
    end

    if test_losses[end] < best_test_loss
        global best_test_loss = test_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(ema_model, joinpath(save_dir, "best"))
    end
end

# save
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "MSE loss")

if !isempty(saved_preds)
    pred_matrix = reduce(hcat, saved_preds)
    target_matrix = reduce(hcat, saved_targets)
    jldsave(joinpath(save_dir, "lrecon_diagnostics.jld2");
        preds=pred_matrix, targets=target_matrix, positions=saved_positions,
        target_variances=target_variances)
end

plot_per_gene_error(gene_error_sums, gene_error_counts, n_coding, save_dir,
                    "mean embedding MSE", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, n_coding, save_dir,
                           "mean embedding MSE", "per_rank_error")

log_model(ema_model, save_dir)

# save shard split for finetune reuse
jldsave(joinpath(save_dir, "shard_split.jld2");
        train_shards=train_shards, test_shards=test_shards)

log_info(; save_dir=save_dir, train_indices=Int[], test_indices=Int[],
           n_epochs=length(train_losses), train_losses=train_losses,
           test_losses=test_losses, target_variances=target_variances)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, total_steps=global_step,
           best_epoch=best_epoch, best_test_loss=best_test_loss)

wb !== nothing && wandb.finish()
println("Done. Best test loss: $(round(best_test_loss, digits=4)) at epoch $best_epoch")

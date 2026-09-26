# SC latent-reconstruction pretraining (data2vec-style EMA teacher; Tahoe single-cell shards, streamed)
#   student sees the masked/corrupted input and regresses the teacher's (batch-standardized) embeddings of the clean input
#   RTF: top-k gene ids by rank, masked ids; ETF: expression at the 1024 HVGs in fixed gene order, donor-swap corruption

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config
using LoadSC, ProcessSC, EvalSC

args = load_pretrain_args()
config = load_config(args["config"], args;
                     hp_section=["pretrain", "lrecon", args["modeltype"]],
                     dataset="tahoe_sc")

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

use_exp = config["modeltype"] == "etf"
start_time = now()

# data
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
train_shards, val_shards, test_shards = split_shards(list_shards(config["data_dir"]), config["subset_shards"])
top_k = config["top_k"]
MASK_ID = Int32(n_coding + 1)
hvg_idx = use_exp ? load_hvg_idx(get(config, "hvg_path", "")) : nothing
seq_len = use_exp ? length(hvg_idx) : top_k

mc = (obj=:lrecon, use_exp=use_exp, seq_len=seq_len, mask_ratio=config["mask_ratio"], mask_id=MASK_ID,
      batch_size=config["batch_size"], coding_tokens=coding_tokens, n_coding=n_coding, top_k=top_k,
      token_to_idx=token_to_idx, hvg_idx=hvg_idx, load_shard_fn=load_shard_pyarrow)

# student + EMA teacher
model = if use_exp
    ExpLReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                   n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                   dropout_prob=config["drop_prob"], seq_len=seq_len)
else
    RankLReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                    dropout_prob=config["drop_prob"], seq_len=seq_len)
end
model = fix_gpu_dropout(cu(model))
ema_model = deepcopy(model)
Flux.testmode!(ema_model)
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)
mask_bufs = sc_mask_buffers(mc)

# save dir + logging
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))
save_dir = joinpath("results", "tahoe", "sc", "pretrain", "lrecon", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")
wandb = init_wandb(config, "SC-PT-Aug", "lrecon_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

# schedule: batches/epoch estimated from one shard, per-step warmup + cosine lr
n_cells_per_shard = load_shard_pyarrow(train_shards[1]).n_cells
sched = step_schedule(n_cells_per_shard, length(train_shards), config["batch_size"], config["max_steps"], config["n_epochs"])
println("  $(n_cells_per_shard) cells/shard -> $(sched.bpe) batches/epoch, $(sched.n_epochs) epochs, " *
        "$(sched.total_steps) steps (warmup $(sched.warmup_steps))")
use_max_steps = config["max_steps"] > 0
MAX_EMBED_BATCHES = 10   # test batches whose raw embeddings are kept in lrecon_diagnostics.jld2

# statically masked val (checkpoint selection, every epoch) and test (final epoch) batches
val_cache = sc_masked_cache(val_shards[1:min(config["n_eval_shards"], length(val_shards))], mc)
eval_cache = sc_masked_cache(test_shards[1:min(config["n_eval_shards"], length(test_shards))], mc)

train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
target_variances = Float32[]   # raw (un-standardized) teacher embedding variance on a fixed batch: collapse monitor
err_acc = ErrorAcc(n_coding, seq_len)
lrecon_diag = LreconDiag()
collapse_check_batch = nothing

global_step = 0
done = false
best_val_loss = Inf32
best_epoch = 0

for epoch in ProgressBar(1:sched.n_epochs)
    done && break
    lr = compute_lr_step(global_step + 1, sched.total_steps, config["lr"], sched.warmup_steps)

    # train
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    shuffled_train = shuffle(train_shards)
    for (si, shard_path) in enumerate(shuffled_train)
        done && break
        for batch in sc_shard_batches(shard_path, mc)
            lr = compute_lr_step(global_step + 1, sched.total_steps, config["lr"], sched.warmup_steps)
            Optimisers.adjust!(opt, lr)
            b = sc_mask!(mask_bufs, batch, mc)
            x_gpu, mask_gpu = CuArray(b.x), CuArray(Float32.(b.m))
            targets = teacher_targets(ema_model, CuArray(b.clean), use_exp)
            l_val, grads = Flux.withgradient(model) do m
                masked_lrecon_loss(m, x_gpu, targets, mask_gpu)[1]
            end
            Flux.update!(opt, model, grads[1])
            ema_update!(ema_model, model, Float32(config["ema_decay"]))
            push!(epoch_losses, l_val)
            isnothing(collapse_check_batch) && (global collapse_check_batch = copy(b.clean))
            global global_step += 1
            if use_max_steps && global_step >= config["max_steps"]
                global done = true; break
            end
        end
        if si % 50 == 0 || done
            println("  epoch $epoch shard $si/$(length(shuffled_train)) step=$global_step loss=$(round(mean(epoch_losses[max(1,end-49):end]), digits=4))")
        end
    end
    push!(train_losses, mean(epoch_losses))

    # collapse monitor: raw teacher variance across cells (standardized targets have var ≈ 1 by construction)
    # raw_tgts = teacher_targets(ema_model, CuArray(collapse_check_batch), use_exp; normalize=false)
    # push!(target_variances, Float32(mean(cpu(var(raw_tgts, dims=3)))))
    # variance across cells over real tokens only (PAD positions of short cells excluded)
    raw_tgts, tgt_keep = teacher_targets(ema_model, CuArray(collapse_check_batch), use_exp; normalize=false, return_keep=true)
    if isnothing(tgt_keep)
        push!(target_variances, Float32(mean(cpu(var(raw_tgts, dims=3)))))
    else
        w = reshape(Float32.(tgt_keep), 1, size(tgt_keep)...)
        n = sum(w, dims=3)
        mu = sum(raw_tgts .* w, dims=3) ./ max.(n, 1f0)
        v = sum(((raw_tgts .- mu) .* w) .^ 2, dims=3) ./ max.(n .- 1f0, 1f0)
        ok_pos = repeat(n .>= 2f0, size(v, 1), 1, 1)
        push!(target_variances, Float32(mean(cpu(v)[cpu(ok_pos)])))
    end
    target_variances[end] < 1f-6 && println("WARNING epoch $epoch: target embedding variance = $(target_variances[end]) — possible collapse")

    # val (every epoch, checkpoint selection; student scored against the current teacher)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    for c in val_cache
        targets = teacher_targets(ema_model, CuArray(c.clean), use_exp)
        push!(val_eval_losses, Float32(masked_lrecon_loss(model, CuArray(c.x), targets, CuArray(Float32.(c.m)))[1]))
    end
    push!(val_losses, mean(val_eval_losses))
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        save_best(model, save_dir, config; ema=ema_model)
    end

    # test (final epoch, best student)
    is_last = (epoch == sched.n_epochs) || done
    if is_last
        best_cpu = load_best_cpu(model, save_dir)
        if !isnothing(best_cpu)
            global model = fix_gpu_dropout(cu(best_cpu))
            Flux.testmode!(model)
            println("reloaded best model (epoch $best_epoch) for test eval")
        end
        eval_losses = Float32[]
        for (bi, c) in enumerate(eval_cache)
            targets = teacher_targets(ema_model, CuArray(c.clean), use_exp)
            loss_val, preds_embed, targets_embed = masked_lrecon_loss(model, CuArray(c.x), targets, CuArray(Float32.(c.m)))
            push!(eval_losses, Float32(loss_val))
            isnothing(preds_embed) && continue
            accumulate_lrecon_diag!(err_acc, lrecon_diag, cpu(preds_embed), cpu(targets_embed), c.m, c.ids_or_ranks, hvg_idx;
                                    save_embed = bi <= MAX_EMBED_BATCHES)
        end
        push!(test_losses, mean(eval_losses))
    end

    println("epoch $epoch/$(sched.n_epochs) | train=$(round(train_losses[end], digits=4)) val=$(round(val_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")
    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end], "val_loss" => val_losses[end],
                        "target_variance" => target_variances[end], "global_step" => global_step, "lr" => lr)
        is_last && (log_dict["test_loss"] = test_losses[end])
        wb.log(log_dict)
    end
end

# plots + outputs
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "MSE loss"; val_losses=val_losses)

if !isempty(lrecon_diag.mse)
    diag_out = Dict{Symbol, Any}(:mse => lrecon_diag.mse, :cossim => lrecon_diag.cossim, :positions => lrecon_diag.positions,
                                 :target_variances => target_variances)
    if !isempty(lrecon_diag.sample_preds)
        diag_out[:sample_preds] = reduce(hcat, lrecon_diag.sample_preds)
        diag_out[:sample_targets] = reduce(hcat, lrecon_diag.sample_targets)
        diag_out[:sample_positions] = lrecon_diag.sample_positions
    end
    jldsave(joinpath(save_dir, "lrecon_diagnostics.jld2"); diag_out...)
end

if any(>(0), err_acc.gene_counts)
    plot_per_gene_error(err_acc.gene_sums, err_acc.gene_counts, n_coding, save_dir,
                        "mean embedding MSE", "per_gene_error";
                        sorted_gene_path=get(config, "sorted_gene_path", ""))
    plot_per_sample_rank_error(err_acc.rank_sums, err_acc.rank_counts, seq_len, save_dir,
                               "mean embedding MSE", "per_rank_error")
else
    println("  skipping per-gene/per-rank plots: no predictions collected")
end

log_model(model, save_dir, config)
use_exp && jldsave(joinpath(save_dir, "hvg_indices.jld2"); hvg_idx=hvg_idx)
mkpath(joinpath(save_dir, "ema"))
log_model(ema_model, joinpath(save_dir, "ema"), config)   # teacher
jldsave(joinpath(save_dir, "shard_split.jld2");   # reused by SC finetuning
        train_shards=train_shards, val_shards=val_shards, test_shards=test_shards)
log_info(; save_dir=save_dir, train_indices=Int[], val_indices=Int[], test_indices=Int[],
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           target_variances=target_variances)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)
log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, total_steps=global_step,
           best_epoch=best_epoch, best_val_loss=best_val_loss)

wb !== nothing && wandb.finish()
println("Done. Best val loss: $(round(best_val_loss, digits=4)) at epoch $best_epoch")

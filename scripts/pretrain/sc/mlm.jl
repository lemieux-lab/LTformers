# SC masked-language-model pretraining (Tahoe single-cell shards, streamed)
#   RTF: top-k gene ids by rank, predict the gene id at masked ranks
#   ETF: expression at the 1024 HVGs in fixed gene order (data/hvg_indices.jld2), predict each masked gene's rank

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
                     hp_section=["pretrain", "mlm", args["modeltype"]],
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
n_classes = use_exp ? seq_len : n_coding   # ETF: rank within the HVG set; RTF: gene id

mc = (obj=:mlm, use_exp=use_exp, seq_len=seq_len, mask_ratio=config["mask_ratio"], mask_id=MASK_ID,
      batch_size=config["batch_size"], coding_tokens=coding_tokens, n_coding=n_coding, top_k=top_k,
      token_to_idx=token_to_idx, hvg_idx=hvg_idx, load_shard_fn=load_shard_pyarrow)

# model
model = if use_exp
    ExpModel(n_genes=seq_len, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
             n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
             dropout_prob=config["drop_prob"], seq_len=seq_len)
else
    RankModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
              n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
              dropout_prob=config["drop_prob"], seq_len=seq_len)
end
model = fix_gpu_dropout(cu(model))
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)
mask_bufs = sc_mask_buffers(mc)

# save dir + logging
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))
save_dir = joinpath("results", "tahoe", "sc", "pretrain", "mlm", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")
wandb = init_wandb(config, "SC-PT-Aug", "mlm_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

# schedule: batches/epoch estimated from one shard, per-step warmup + cosine lr
n_cells_per_shard = load_shard_pyarrow(train_shards[1]).n_cells
sched = step_schedule(n_cells_per_shard, length(train_shards), config["batch_size"], config["max_steps"], config["n_epochs"])
println("  $(n_cells_per_shard) cells/shard -> $(sched.bpe) batches/epoch, $(sched.n_epochs) epochs, " *
        "$(sched.total_steps) steps (warmup $(sched.warmup_steps))")
use_max_steps = config["max_steps"] > 0
max_pred_batches = 5 * cld(n_cells_per_shard, config["batch_size"])   # cap predstrues at ~5 shards

# statically masked val (checkpoint selection, every epoch) and test (final epoch) batches
val_cache = sc_masked_cache(val_shards[1:min(config["n_eval_shards"], length(val_shards))], mc)
eval_cache = sc_masked_cache(test_shards[1:min(config["n_eval_shards"], length(test_shards))], mc)

train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]
all_preds = Int[]
all_trues = Int[]
err_acc = ErrorAcc(n_coding, seq_len)

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
            x_gpu, y_gpu = CuArray(b.x), CuArray(b.y)
            l_val, grads = Flux.withgradient(model) do m
                sc_masked_loss(m, x_gpu, y_gpu, n_classes)[1]
            end
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, l_val)
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

    # val (every epoch, checkpoint selection)
    Flux.testmode!(model)
    push!(val_losses, mean(Float32(sc_masked_loss(model, CuArray(c.x), CuArray(c.y), n_classes)[1]) for c in val_cache))
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        save_best(model, save_dir, config)
    end

    # test (final epoch, on the best checkpoint)
    is_last = (epoch == sched.n_epochs) || done
    if is_last
        best_cpu = load_best_cpu(model, save_dir)
        if !isnothing(best_cpu)
            global model = fix_gpu_dropout(cu(best_cpu))
            Flux.testmode!(model)
            println("reloaded best model (epoch $best_epoch) for test eval")
        end
        eval_losses = Float32[]
        epoch_rank_errors = Int[]
        n_pred_batches = 0
        for c in eval_cache
            loss_val, logits_masked, y_targets = sc_masked_loss(model, CuArray(c.x), CuArray(c.y), n_classes)
            push!(eval_losses, Float32(loss_val))
            isnothing(y_targets) && continue
            errs = gpu_rank_errors(logits_masked, y_targets)
            append!(epoch_rank_errors, errs)
            accumulate_rank_errors!(err_acc, errs, c.y, n_classes, hvg_idx)
            if n_pred_batches < max_pred_batches
                append!(all_preds, Flux.onecold(cpu(logits_masked)))
                append!(all_trues, cpu(y_targets))
                n_pred_batches += 1
            end
        end
        push!(test_losses, mean(eval_losses))
        push!(test_rank_errors, isempty(epoch_rank_errors) ? NaN32 : mean(Float32.(epoch_rank_errors)))
        println("  predstrues: $n_pred_batches batches ($(length(all_preds)) predictions)")
    end

    println("epoch $epoch/$(sched.n_epochs) | train=$(round(train_losses[end], digits=4)) val=$(round(val_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")
    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end], "val_loss" => val_losses[end],
                        "global_step" => global_step, "lr" => lr)
        if is_last
            log_dict["test_loss"] = test_losses[end]
            log_dict["mean_rank_error"] = test_rank_errors[end]
        end
        wb.log(log_dict)
    end
end

# plots + outputs
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "logit-ce"; val_losses=val_losses)
cs, cp = if !isempty(all_preds)
    plot_ranked_heatmap(all_trues, all_preds, save_dir)
else
    println("  skipping heatmap: no predictions collected")
    (NaN, NaN)
end
plot_per_gene_error(err_acc.gene_sums, err_acc.gene_counts, n_coding, save_dir,
                    "mean rank error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(err_acc.rank_sums, err_acc.rank_counts, seq_len, save_dir,
                           "mean rank error", "per_rank_error")

log_model(model, save_dir, config)
use_exp && jldsave(joinpath(save_dir, "hvg_indices.jld2"); hvg_idx=hvg_idx)
jldsave(joinpath(save_dir, "shard_split.jld2");   # reused by SC finetuning
        train_shards=train_shards, val_shards=val_shards, test_shards=test_shards)
log_info(; save_dir=save_dir, train_indices=Int[], val_indices=Int[], test_indices=Int[],
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)
log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, pearson=cp, spearman=cs,
           total_steps=global_step, best_epoch=best_epoch,
           best_val_loss=best_val_loss)

wb !== nothing && wandb.finish()
println("Done. Best val loss: $(round(best_val_loss, digits=4)) at epoch $best_epoch")

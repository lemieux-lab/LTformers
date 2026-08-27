using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics, Functors
using ProgressBars, CairoMakie, StatsBase

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using Preprocess, Models, Train, Log, Plot, Args, Config
using LoadSC, ProcessSC
@eval ProcessSC using Models: encode

args = load_pretrain_args()
config = load_config(args["config"], args)

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

use_exp = config["modeltype"] == "etf"

start_time = now()
_t0 = time()
_elapsed() = round(time() - _t0, digits=1)
_phase(name) = (println("[$(round(time() - _t0, digits=1))s] $name"); flush(stdout))

_phase("load_gene_vocab()")
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])

# _phase("list_shards() + shard_train_test_split()")
_phase("list_shards() + shard_train_val_test_split()")
all_shards = list_shards(config["data_dir"])
# train_shards, test_shards = shard_train_test_split(all_shards, 0.2)
train_shards, val_shards, test_shards = shard_train_val_test_split(all_shards, 0.1, 0.1)
if config["subset_shards"] > 0
    train_shards = train_shards[1:min(config["subset_shards"], length(train_shards))]
    val_shards = val_shards[1:min(max(1, div(config["subset_shards"], 8)), length(val_shards))]
    test_shards = test_shards[1:min(max(1, div(config["subset_shards"], 8)), length(test_shards))]
end
# println("Shards: $(length(train_shards)) train, $(length(test_shards)) test")
println("Shards: $(length(train_shards)) train, $(length(val_shards)) val, $(length(test_shards)) test")

top_k = config["top_k"]
n_classes = n_coding
MASK_ID = Int32(n_coding + 1)

_phase("$(use_exp ? "ExpModel" : "RankModel")()")
model = if use_exp
    ExpModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
             n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
             dropout_prob=config["drop_prob"], seq_len=top_k)
else
    RankModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
              n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
              dropout_prob=config["drop_prob"], seq_len=top_k)
end
_phase("cu() + fix_gpu_dropout()")
model = cu(model)
model = fix_gpu_dropout(model)

# opt = Flux.setup(Adam(config["lr"]), model)
# opt = Flux.setup(AdamW(config["lr"]), model)
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)

# masking buffers
X_masked_rtf = Matrix{Int32}(undef, top_k, config["batch_size"])
y_masked_rtf = Matrix{Int32}(undef, top_k, config["batch_size"])
X_masked_etf = Matrix{Float32}(undef, top_k, config["batch_size"])
y_masked_etf = Matrix{Int32}(undef, top_k, config["batch_size"])

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("results", "tahoe", "sc", "pretrain", "mlm", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

_phase("init_wandb()")
wandb = init_wandb(config, "SC-PT-Aug", "mlm_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]
all_preds = Int[]
all_trues = Int[]
# rank_error_sums = zeros(Float32, n_coding)
# rank_error_counts = zeros(Int, n_coding)
rank_error_sums = zeros(Float32, top_k)   # indexed by rank position (1..top_k)
rank_error_counts = zeros(Int, top_k)
gene_error_sums = zeros(Float32, n_coding)  # indexed by gene_id (1..n_coding)
gene_error_counts = zeros(Int, n_coding)

# pre-allocate GPU buffers (#4) — reverted: copyto! from CPU SubArray to CuArray
# triggers scalar indexing (CUDA.jl doesn't specialize for SubArray sources)
# x_gpu_buf_rtf = CUDA.zeros(Int32, top_k, config["batch_size"])
# y_gpu_buf_rtf = CUDA.zeros(Int32, top_k, config["batch_size"])
# x_gpu_buf_etf = CUDA.zeros(Float32, top_k, config["batch_size"])
# y_gpu_buf_etf = CUDA.zeros(Int32, top_k, config["batch_size"])

# cap predstrues to avoid multi-GB files (#8)
const MAX_PRED_SHARDS = 5

global_step = 0
use_max_steps = config["max_steps"] > 0
done = false
# best_test_loss = Inf32
best_val_loss = Inf32
best_epoch = 0

_phase("load_shard_pyarrow() — estimating epochs")
n_total_epochs = if use_max_steps
    # sample 1 shard instead of 5 to avoid slow PyArrow startup overhead
    # n_sample = min(5, length(train_shards))
    n_sample = 1
    sample_shards = train_shards[1:n_sample]
    local total_cells = 0
    for sp in sample_shards
        println("  estimating epoch size from shard: $(basename(sp))")
        shard = load_shard_pyarrow(sp)
        total_cells += shard.n_cells
        println("  shard has $(shard.n_cells) cells")
    end
    avg_cells_per_shard = total_cells / n_sample
    est_cells_per_epoch = avg_cells_per_shard * length(train_shards)
    bpe = cld(Int(round(est_cells_per_epoch)), config["batch_size"])
    n_ep = cld(config["max_steps"], bpe)
    println("  estimated: $(Int(round(avg_cells_per_shard))) cells/shard, $bpe batches/epoch, $n_ep epochs for $(config["max_steps"]) steps")
    n_ep
else
    config["n_epochs"]
end
warmup_epochs = max(1, div(n_total_epochs, 10))

# pre-cache val batches with static masks (used every epoch for checkpoint selection)
_phase("pre-caching val batches with static masks")
val_cache = NamedTuple[]
_val_shards = val_shards[1:min(config["n_eval_shards"], length(val_shards))]
for sp in _val_shards
    batches = batches_from_shard(sp, coding_tokens, n_coding, top_k,
                                 config["batch_size"]; modeltype=config["modeltype"],
                                 token_to_idx=token_to_idx,
                                 load_shard_fn=load_shard_pyarrow)
    for batch in batches
        if use_exp
            batch_ids, batch_expr = batch
            bs = size(batch_expr, 2)
            xm = Matrix{Float32}(undef, top_k, bs)
            ym = Matrix{Int32}(undef, top_k, bs)
            sc_mask_input_exp!(xm, ym, batch_expr, batch_ids, config["mask_ratio"], -100)
            push!(val_cache, (x=copy(xm), y=copy(ym), ids=copy(batch_ids)))
        else
            bs = size(batch, 2)
            xm = Matrix{Int32}(undef, top_k, bs)
            ym = Matrix{Int32}(undef, top_k, bs)
            sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
            push!(val_cache, (x=copy(xm), y=copy(ym), raw=copy(batch)))
        end
    end
end
println("  cached $(length(val_cache)) val batches from $(length(_val_shards)) shards")

# pre-cache test batches with static masks (used only on final epoch)
_phase("pre-caching test batches with static masks")
eval_cache = NamedTuple[]
_eval_shards = test_shards[1:min(config["n_eval_shards"], length(test_shards))]
for sp in _eval_shards
    batches = batches_from_shard(sp, coding_tokens, n_coding, top_k,
                                 config["batch_size"]; modeltype=config["modeltype"],
                                 token_to_idx=token_to_idx,
                                 load_shard_fn=load_shard_pyarrow)
    for batch in batches
        if use_exp
            batch_ids, batch_expr = batch
            bs = size(batch_expr, 2)
            xm = Matrix{Float32}(undef, top_k, bs)
            ym = Matrix{Int32}(undef, top_k, bs)
            sc_mask_input_exp!(xm, ym, batch_expr, batch_ids, config["mask_ratio"], -100)
            push!(eval_cache, (x=copy(xm), y=copy(ym), ids=copy(batch_ids)))
        else
            bs = size(batch, 2)
            xm = Matrix{Int32}(undef, top_k, bs)
            ym = Matrix{Int32}(undef, top_k, bs)
            sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
            push!(eval_cache, (x=copy(xm), y=copy(ym), raw=copy(batch)))
        end
    end
end
println("  cached $(length(eval_cache)) test batches from $(length(_eval_shards)) shards")

_phase("training loop — batches_from_shard() + sc_masked_loss() + Flux.withgradient()")
# for epoch in 1:n_total_epochs
for epoch in ProgressBar(1:n_total_epochs)
    done && break
    lr = compute_lr(epoch, n_total_epochs, config["lr"], warmup_epochs)
    Optimisers.adjust!(opt, lr)

    # train
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    shuffled_train = shuffle(train_shards)

    for (si, shard_path) in enumerate(shuffled_train)
        done && break
        shard_t0 = time()

        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype=config["modeltype"],
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)

        n_shard_steps = 0
        for batch in batches
            if use_exp
                batch_ids, batch_expr = batch
                bs = size(batch_expr, 2)
                xm = @view X_masked_etf[:, 1:bs]
                ym = @view y_masked_etf[:, 1:bs]
                sc_mask_input_exp!(xm, ym, batch_expr, batch_ids, config["mask_ratio"], -100)
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)
            else
                bs = size(batch, 2)
                xm = @view X_masked_rtf[:, 1:bs]
                ym = @view y_masked_rtf[:, 1:bs]
                sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)
            end

            l_val, grads = Flux.withgradient(model) do m
                sc_masked_loss(m, x_gpu, y_gpu, n_classes)[1]
            end
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, l_val)
            n_shard_steps += 1
            global global_step += 1
            if use_max_steps && global_step >= config["max_steps"]
                global done = true; break
            end
        end

        shard_dt = round(time() - shard_t0, digits=1)
        if si <= 3 || si % 50 == 0 || done
            println("  [$(round(time() - _t0, digits=1))s] epoch $epoch shard $si/$(length(shuffled_train)) steps=$n_shard_steps ($(shard_dt)s) step=$global_step loss=$(round(mean(epoch_losses[max(1,end-49):end]), digits=4))")
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval (every epoch — used for checkpoint selection)
    _phase("val eval (epoch $epoch)")
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    for cached in val_cache
        x_gpu = CuArray(cached.x)
        y_gpu = CuArray(cached.y)
        loss_val, _, _ = sc_masked_loss(model, x_gpu, y_gpu, n_classes)
        push!(val_eval_losses, loss_val)
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only — held out for reporting)
    is_last = (epoch == n_total_epochs) || done
    eval_losses = Float32[]
    epoch_rank_errors = Int[]
    epoch_preds = Int[]
    epoch_trues = Int[]

    if is_last
        full_eval = !use_max_steps
        pred_shard_count = 0

        if full_eval
            _phase("full test eval (epoch $epoch)")
            for shard_path in test_shards
                batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                             config["batch_size"]; modeltype=config["modeltype"],
                                             token_to_idx=token_to_idx,
                                             load_shard_fn=load_shard_pyarrow)
                collect_preds = pred_shard_count < MAX_PRED_SHARDS

                for batch in batches
                    if use_exp
                        batch_ids, batch_expr = batch
                        bs = size(batch_expr, 2)
                        xm = @view X_masked_etf[:, 1:bs]
                        ym = @view y_masked_etf[:, 1:bs]
                        sc_mask_input_exp!(xm, ym, batch_expr, batch_ids, config["mask_ratio"], -100)
                        x_gpu = CuArray(xm)
                        y_gpu = CuArray(ym)
                    else
                        bs = size(batch, 2)
                        xm = @view X_masked_rtf[:, 1:bs]
                        ym = @view y_masked_rtf[:, 1:bs]
                        sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
                        x_gpu = CuArray(xm)
                        y_gpu = CuArray(ym)
                    end
                    loss_val, logits_masked, y_targets = sc_masked_loss(model, x_gpu, y_gpu, n_classes)
                    push!(eval_losses, loss_val)

                    if !isnothing(y_targets)
                        errs = gpu_rank_errors(logits_masked, y_targets)
                        append!(epoch_rank_errors, errs)

                        if collect_preds
                            logits_cpu = cpu(logits_masked)
                            append!(epoch_preds, Flux.onecold(logits_cpu))
                            append!(epoch_trues, cpu(y_targets))
                        end

                        y_labels_cpu = cpu(y_gpu)
                        batch_len = size(y_labels_cpu, 2)
                        masked_idx = 0
                        for j in 1:batch_len
                            for pos in 1:top_k
                                r = y_labels_cpu[pos, j]
                                (r == -100 || r <= 0 || r > n_classes) && continue
                                masked_idx += 1
                                err = errs[masked_idx]
                                gene_error_sums[r] += err
                                gene_error_counts[r] += 1
                                rank_error_sums[pos] += err
                                rank_error_counts[pos] += 1
                            end
                        end
                    end
                end
                if collect_preds
                    pred_shard_count += 1
                end
            end
        else
            _phase("cached test eval (epoch $epoch)")
            n_cached_preds = 0
            max_cached_preds = MAX_PRED_SHARDS * cld(28225, config["batch_size"])
            for cached in eval_cache
                x_gpu = CuArray(cached.x)
                y_gpu = CuArray(cached.y)
                loss_val, logits_masked, y_targets = sc_masked_loss(model, x_gpu, y_gpu, n_classes)
                push!(eval_losses, loss_val)

                if !isnothing(y_targets)
                    errs = gpu_rank_errors(logits_masked, y_targets)
                    append!(epoch_rank_errors, errs)

                    if n_cached_preds < max_cached_preds
                        logits_cpu = cpu(logits_masked)
                        append!(epoch_preds, Flux.onecold(logits_cpu))
                        append!(epoch_trues, cpu(y_targets))
                        n_cached_preds += 1
                    end
                end
            end
        end

        if !isempty(eval_losses)
            push!(test_losses, mean(eval_losses))
            push!(test_rank_errors, isempty(epoch_rank_errors) ? NaN32 : mean(Float32.(epoch_rank_errors)))
        end

        if full_eval
            append!(all_preds, epoch_preds)
            append!(all_trues, epoch_trues)
            println("  predstrues: $(pred_shard_count) shards collected ($(length(all_preds)) predictions, capped at $MAX_PRED_SHARDS)")
        elseif !isempty(epoch_preds)
            append!(all_preds, epoch_preds)
            append!(all_trues, epoch_trues)
            println("  predstrues: $(n_cached_preds) cached batches collected ($(length(all_preds)) predictions)")
        end
    end

    # println("epoch $epoch/$n_total_epochs | train=$(round(train_losses[end], digits=4)) test=$(round(test_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")
    println("epoch $epoch/$n_total_epochs | train=$(round(train_losses[end], digits=4)) val=$(round(val_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                        "val_loss" => val_losses[end],
                        "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
            log_dict["mean_rank_error"] = test_rank_errors[end]
        end
        wb.log(log_dict)
    end

    # if test_losses[end] < best_test_loss
    #     global best_test_loss = test_losses[end]
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(model, joinpath(save_dir, "best"), config)
    end
end

_phase("plot_loss() + log_model() + log_params()")
# save
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "logit-ce")
cs, cp = if !isempty(all_preds)
    plot_ranked_heatmap(all_trues, all_preds, save_dir)
else
    println("  skipping heatmap: no predictions collected")
    (NaN, NaN)
end
plot_per_rank_error(rank_error_sums, rank_error_counts, top_k, save_dir)
plot_per_gene_error(gene_error_sums, gene_error_counts, n_coding, save_dir,
                    "mean rank error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, top_k, save_dir,
                           "mean rank error", "per_rank_error")

# log_model(model, save_dir)
log_model(model, save_dir, config)
# save shard split for finetune reuse
jldsave(joinpath(save_dir, "shard_split.jld2");
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
_phase("done.")
# println("Done. Best test loss: $(round(best_test_loss, digits=4)) at epoch $best_epoch")
println("Done. Best val loss: $(round(best_val_loss, digits=4)) at epoch $best_epoch")

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

coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])

all_shards = list_shards(config["data_dir"])
train_shards, test_shards = shard_train_test_split(all_shards, 0.2)
println("Shards: $(length(train_shards)) train, $(length(test_shards)) test")

top_k = config["top_k"]
n_classes = n_coding
MASK_ID = Int32(n_coding + 1)

model = if use_exp
    ExpModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
             n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
             dropout_prob=config["drop_prob"], seq_len=top_k)
else
    RankModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
              n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
              dropout_prob=config["drop_prob"], seq_len=top_k)
end
model = cu(model)
model = fix_gpu_dropout(model)

opt = Flux.setup(Adam(config["lr"]), model)

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

wandb = init_wandb(config, "SC-PT-Aug", "mlm_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]
all_preds = Int[]
all_trues = Int[]
rank_error_sums = zeros(Float32, n_coding)
rank_error_counts = zeros(Int, n_coding)
gene_error_sums = zeros(Float32, n_coding)
gene_error_counts = zeros(Int, n_coding)

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
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype=config["modeltype"],
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)

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

    # eval
    Flux.testmode!(model)
    eval_losses = Float32[]
    epoch_rank_errors = Int[]
    epoch_preds = Int[]
    epoch_trues = Int[]
    is_last = (epoch == n_total_epochs) || done
    # eval all test shards on last epoch, subset otherwise
    eval_shards = if is_last
        test_shards
    else
        shuffle(test_shards)[1:min(config["n_eval_shards"], length(test_shards))]
    end

    for shard_path in eval_shards
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
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
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)
            else
                bs = size(batch, 2)
                xm = Matrix{Int32}(undef, top_k, bs)
                ym = Matrix{Int32}(undef, top_k, bs)
                sc_mask_input!(xm, ym, batch, config["mask_ratio"], -100, MASK_ID)
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)
            end
            loss_val, logits_masked, y_targets = sc_masked_loss(model, x_gpu, y_gpu, n_classes)
            push!(eval_losses, loss_val)

            if !isnothing(y_targets)
                logits_cpu = cpu(logits_masked)
                y_targets_cpu = cpu(y_targets)

                if is_last
                    append!(epoch_preds, Flux.onecold(logits_cpu))
                    append!(epoch_trues, y_targets_cpu)
                end

                for i in eachindex(y_targets_cpu)
                    r = y_targets_cpu[i]
                    col = @view logits_cpu[:, i]
                    err = count(x -> x > col[r], col)
                    push!(epoch_rank_errors, err)
                    if is_last
                        rank_error_sums[r] += err
                        rank_error_counts[r] += 1
                    end
                end

                if is_last
                    y_labels_cpu = cpu(y_gpu)
                    batch_len = size(y_labels_cpu, 2)
                    masked_idx = 0
                    for j in 1:batch_len
                        for pos in 1:top_k
                            r = y_labels_cpu[pos, j]
                            (r == -100 || r <= 0 || r > n_classes) && continue
                            masked_idx += 1
                            col = @view logits_cpu[:, masked_idx]
                            err = count(x -> x > col[r], col)
                            gene_error_sums[pos] += err
                            gene_error_counts[pos] += 1
                        end
                    end
                end
            end
        end
    end
    push!(test_losses, mean(eval_losses))
    push!(test_rank_errors, isempty(epoch_rank_errors) ? NaN32 : mean(Float32.(epoch_rank_errors)))

    if is_last
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    println("epoch $epoch/$n_total_epochs | train=$(round(train_losses[end], digits=4)) test=$(round(test_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")

    if wb !== nothing
        wb.log(Dict("epoch" => epoch, "train_loss" => train_losses[end],
                     "test_loss" => test_losses[end],
                     "mean_rank_error" => test_rank_errors[end],
                     "global_step" => global_step))
    end

    if test_losses[end] < best_test_loss
        global best_test_loss = test_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(model, joinpath(save_dir, "best"))
    end
end

# save
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "logit-ce")
cs, cp = plot_ranked_heatmap(all_trues, all_preds, save_dir)
plot_per_rank_error(rank_error_sums, rank_error_counts, n_coding, save_dir)
plot_per_gene_error(gene_error_sums, gene_error_counts, n_coding, save_dir,
                    "mean rank error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, n_coding, save_dir,
                           "mean rank error", "per_rank_error")

log_model(model, save_dir)
# save shard split for finetune reuse
jldsave(joinpath(save_dir, "shard_split.jld2");
        train_shards=train_shards, test_shards=test_shards)

log_info(; save_dir=save_dir, train_indices=Int[], test_indices=Int[],
           n_epochs=length(train_losses), train_losses=train_losses,
           test_losses=test_losses, all_preds=all_preds, all_trues=all_trues)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, pearson=cp, spearman=cs,
           total_steps=global_step, best_epoch=best_epoch,
           best_test_loss=best_test_loss)

wb !== nothing && wandb.finish()
println("Done. Best test loss: $(round(best_test_loss, digits=4)) at epoch $best_epoch")

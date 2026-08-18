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

coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])

all_shards = list_shards(config["data_dir"])
train_shards, test_shards = shard_train_test_split(all_shards, 0.2)
println("Shards: $(length(train_shards)) train, $(length(test_shards)) test")

top_k = config["top_k"]
MASK_ID = Int32(n_coding + 1)

model = if use_exp
    ExpEReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                   n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                   dropout_prob=config["drop_prob"])
else
    RankEReconModel(n_genes=n_coding, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                    dropout_prob=config["drop_prob"])
end
model = cu(model)
model = fix_gpu_dropout(model)

# opt = Flux.setup(Adam(config["lr"]), model)
opt = Flux.setup(AdamW(config["lr"]), model)

# mask
X_masked_rtf = Matrix{Int32}(undef, top_k, config["batch_size"])
y_masked_rtf = Matrix{Float32}(undef, top_k, config["batch_size"])
X_masked_etf = Matrix{Float32}(undef, top_k, config["batch_size"])
corrupt_mask_buf = falses(top_k, config["batch_size"])

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("results", "tahoe", "sc", "pretrain", "erecon", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

wandb = init_wandb(config, "SC-PT-Aug", "erecon_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

train_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]
gene_error_sums = zeros(Float32, n_coding)  # indexed by gene_id (1..n_coding)
gene_error_counts = zeros(Int, n_coding)
# rank_error_sums = zeros(Float32, n_coding)
rank_error_sums = zeros(Float32, top_k)   # indexed by rank position (1..top_k)
rank_error_counts = zeros(Int, top_k)

global_step = 0
use_max_steps = config["max_steps"] > 0
done = false
best_test_loss = Inf32
best_epoch = 0

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
        # get both IDs and expression for erecon
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype="etf",
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)

        for (batch_ids, batch_expr) in batches
            bs = size(batch_expr, 2)

            if use_exp
                xm = @view X_masked_etf[:, 1:bs]
                cm = @view corrupt_mask_buf[:, 1:bs]
                corrupt_expr!(xm, cm, batch_expr, config["mask_ratio"])
                x_gpu = CuArray(xm)
                y_gpu = CuArray(batch_expr)
                m_gpu = CuArray(Float32.(cm))

                l_val, grads = Flux.withgradient(model) do m
                    masked_erecon_loss(m, x_gpu, y_gpu, m_gpu)[1]
                end
            else
                xm = @view X_masked_rtf[:, 1:bs]
                ym = @view y_masked_rtf[:, 1:bs]
                sc_mask_input_erecon!(xm, ym, batch_ids, batch_expr,
                                     config["mask_ratio"], -100f0, MASK_ID)
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)

                l_val, grads = Flux.withgradient(model) do m
                    masked_erecon_loss(m, x_gpu, y_gpu)[1]
                end
            end
            # Flux.update!(opt, model, grads[1])
            grads_clipped = Flux.clipnorm(grads[1], 1.0)
            Flux.update!(opt, model, grads_clipped)
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
    is_last = (epoch == n_total_epochs) || done
    # eval all test shards on last epoch, subset otherwise
    eval_shards = if is_last
        test_shards
    else
        shuffle(test_shards)[1:min(config["n_eval_shards"], length(test_shards))]
    end

    for shard_path in eval_shards
        batches = batches_from_shard(shard_path, coding_tokens, n_coding, top_k,
                                     config["batch_size"]; modeltype="etf",
                                     token_to_idx=token_to_idx,
                                     load_shard_fn=load_shard_pyarrow)
        for (batch_ids, batch_expr) in batches
            bs = size(batch_expr, 2)

            if use_exp
                xm = Matrix{Float32}(undef, top_k, bs)
                cm = falses(top_k, bs)
                corrupt_expr!(xm, cm, batch_expr, config["mask_ratio"])
                x_gpu = CuArray(xm)
                y_gpu = CuArray(batch_expr)
                m_gpu = CuArray(Float32.(cm))
                loss_val, preds_masked, targets_masked = masked_erecon_loss(model, x_gpu, y_gpu, m_gpu)
            else
                xm = Matrix{Int32}(undef, top_k, bs)
                ym = Matrix{Float32}(undef, top_k, bs)
                sc_mask_input_erecon!(xm, ym, batch_ids, batch_expr,
                                     config["mask_ratio"], -100f0, MASK_ID)
                x_gpu = CuArray(xm)
                y_gpu = CuArray(ym)
                loss_val, preds_masked, targets_masked = masked_erecon_loss(model, x_gpu, y_gpu)
            end
            push!(eval_losses, cpu(loss_val))

            if is_last && !isnothing(preds_masked)
                preds_cpu = vec(cpu(preds_masked))
                targets_cpu = vec(cpu(targets_masked))
                append!(all_preds, preds_cpu)
                append!(all_trues, targets_cpu)

                # per-gene and per-rank errors
                if use_exp
                    mask_cpu_bool = cpu(m_gpu) .> 0f0
                else
                    y_labels_cpu = cpu(y_gpu)
                end
                masked_idx = 0
                for j in 1:bs
                    for pos in 1:top_k
                        if use_exp
                            mask_cpu_bool[pos, j] || continue
                        else
                            (y_labels_cpu[pos, j] == -100f0) && continue
                        end
                        masked_idx += 1
                        err = (preds_cpu[masked_idx] - targets_cpu[masked_idx])^2
                        # pos = rank position (SC data is sorted by expression)
                        rank_error_sums[pos] += err
                        rank_error_counts[pos] += 1
                        gene_id = batch_ids[pos, j]  # gene at rank pos
                        gene_error_sums[gene_id] += err
                        gene_error_counts[gene_id] += 1
                    end
                end
            end
        end
    end
    push!(test_losses, mean(eval_losses))

    println("epoch $epoch/$n_total_epochs | train=$(round(train_losses[end], digits=4)) test=$(round(test_losses[end], digits=4)) steps=$global_step lr=$(round(lr, sigdigits=3))")

    if wb !== nothing
        wb.log(Dict("epoch" => epoch, "train_loss" => train_losses[end],
                     "test_loss" => test_losses[end], "global_step" => global_step))
    end

    if test_losses[end] < best_test_loss
        global best_test_loss = test_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(model, joinpath(save_dir, "best"), config)
    end
end

# save
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "MSE")

# correlation scatter
using StatsBase: corspearman
cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

fig_scatter = Figure(size=(600, 400))
ax_scatter = Axis(fig_scatter[1, 1], xlabel="True", ylabel="Predicted", title="erecon scatter")
n_plot = min(50000, length(all_trues))
plot_idx = sample(1:length(all_trues), n_plot, replace=false)
scatter!(ax_scatter, all_trues[plot_idx], all_preds[plot_idx], markersize=1, alpha=0.3)
text!(ax_scatter, 0.05, 0.95, space=:relative, align=(:left, :top),
    text="Pearson: $(round(cp, digits=4))\nSpearman: $(round(cs, digits=4))")
save(joinpath(save_dir, "scatter.png"), fig_scatter)

plot_per_gene_error(gene_error_sums, gene_error_counts, n_coding, save_dir,
                    "mean squared error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, top_k, save_dir,
                           "mean squared error", "per_rank_error")

# log_model(model, save_dir)
log_model(model, save_dir, config)

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

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

# gene selection: ETF = HVG (gene-position paradigm), RTF = no HVG (rank-position paradigm)
# n_hvg = get(config, "n_hvg", 0)
# if n_hvg > 0 && n_hvg < size(data_expr, 1)
#     n_orig = size(data_expr, 1)
#     data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
#     println("HVG filter: $(n_orig) → $(n_hvg) genes")
# end
if use_exp  # ETF: HVG selects fixed gene set; position encodes gene identity
    n_hvg = get(config, "n_hvg", 0)
    if n_hvg > 0 && n_hvg < size(data_expr, 1)
        n_orig = size(data_expr, 1)
        data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
        println("HVG filter: $(n_orig) → $(n_hvg) genes")
    end
else  # RTF: no HVG — needs full gene vocab for embedding lookup
    println("RTF: using full gene vocab ($(size(data_expr, 1)) genes), top_k truncation")
end

gene_medians = vec(median(data_expr, dims=2)) .+ 1f-10
X_ranks = rank_genes(data_expr, gene_medians)

n_genes = size(X_ranks, 1)
n_classes = n_genes
MASK_ID = n_genes + 1
top_k = get(config, "top_k", 1024)

# _, _, train_indices, test_indices = ttsplit(X_ranks, 0.2f0)
_, _, _, train_indices, val_indices, test_indices = tvsplit(X_ranks, 0.1f0, 0.1f0)

if use_exp
    X_expr = Float32.(data_expr)
    # # OLD: reindex expression into rank order — predicts gene ID at rank (doesn't learn well)
    # X_expr_ranked = reindex_to_rank_order(X_expr, X_ranks)
    # X_train = X_expr_ranked[:, train_indices]
    # X_test = X_expr_ranked[:, test_indices]
    # X_ranks_train = X_ranks[:, train_indices]
    # X_ranks_test = X_ranks[:, test_indices]
    # gene-ordered expression: position i = gene i, labels = rank of each gene
    X_train = X_expr[:, train_indices]
    X_val = X_expr[:, val_indices]
    X_test = X_expr[:, test_indices]
    inv_ranks = inverse_ranks(X_ranks)
    X_inv_ranks_train = inv_ranks[:, train_indices]
    X_inv_ranks_val = inv_ranks[:, val_indices]
    X_inv_ranks_test = inv_ranks[:, test_indices]
else
    # RTF: truncate to top_k (row 1 = highest expressed gene's ID)
    X_ranks_full = X_ranks
    if top_k < n_genes
        X_ranks = X_ranks[1:top_k, :]
        println("top_k truncation: $(n_genes) → $top_k ranked genes per sample")
    end
    X_train = X_ranks[:, train_indices]
    X_val = X_ranks[:, val_indices]
    X_test = X_ranks[:, test_indices]
end

seq_len = use_exp ? n_genes : min(top_k, n_genes)

model = if use_exp
    ExpModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
             n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
             dropout_prob=config["drop_prob"], seq_len=seq_len)
else
    RankModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
              n_classes=n_classes, n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
              dropout_prob=config["drop_prob"], seq_len=seq_len)
end
model = cu(model)
model = fix_gpu_dropout(model)

# opt = Flux.setup(Adam(config["lr"]), model)
# opt = Flux.setup(AdamW(config["lr"]), model)
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)

# mask val
X_val_masked = similar(X_val)
y_val_masked = use_exp ? similar(X_inv_ranks_val) : similar(X_val)
if use_exp
    mask_input_exp!(X_val_masked, y_val_masked, X_val, X_inv_ranks_val, config["mask_ratio"], -100)
else
    mask_input!(X_val_masked, y_val_masked, X_val, config["mask_ratio"], -100, MASK_ID, false)
end

# mask test
X_test_masked = similar(X_test)
# y_test_masked = use_exp ? similar(X_ranks_test) : similar(X_test)
y_test_masked = use_exp ? similar(X_inv_ranks_test) : similar(X_test)
if use_exp
    # mask_input_exp!(X_test_masked, y_test_masked, X_test, X_ranks_test, config["mask_ratio"], -100)
    mask_input_exp!(X_test_masked, y_test_masked, X_test, X_inv_ranks_test, config["mask_ratio"], -100)
else
    mask_input!(X_test_masked, y_test_masked, X_test, config["mask_ratio"], -100, MASK_ID, false)
end

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
fmt = get(config, "data_format", "tahoe")
dataset_tag = fmt == "lincs" ? joinpath("lincs") : joinpath("tahoe", "pb")
save_dir = joinpath("results", dataset_tag, "pretrain", "mlm", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")
if use_exp && @isdefined(hvg_idx)
    jldsave(joinpath(save_dir, "hvg_indices.jld2"); hvg_idx=hvg_idx)
end

wandb = init_wandb(config, "PB-PT-Aug", "mlm_$(fmt)_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

# train
train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]
all_preds = Int[]
all_trues = Int[]
rank_error_sums = zeros(Float32, n_genes)
rank_error_counts = zeros(Int, n_genes)
gene_error_sums = zeros(Float32, n_genes)
gene_error_counts = zeros(Int, n_genes)

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
# y_train_masked = use_exp ? similar(X_ranks_train) : similar(X_train)
y_train_masked = use_exp ? similar(X_inv_ranks_train) : similar(X_train)

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)
    Optimisers.adjust!(opt, compute_lr(epoch, n_total_epochs, config["lr"], warmup_epochs))

    if use_exp
        # mask_input_exp!(X_train_masked, y_train_masked, X_train, X_ranks_train, config["mask_ratio"], -100)
        mask_input_exp!(X_train_masked, y_train_masked, X_train, X_inv_ranks_train, config["mask_ratio"], -100)
    else
        mask_input!(X_train_masked, y_train_masked, X_train, config["mask_ratio"], -100, MASK_ID, false)
    end

    # train epoch
    Flux.trainmode!(model)
    epoch_losses = Float32[]

    for start_idx in 1:config["batch_size"]:size(X_train_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_train_masked, 2))

        x_batch = CuArray(X_train_masked[:, start_idx:end_idx])
        y_batch = CuArray(y_train_masked[:, start_idx:end_idx])

        l_val, grads = Flux.withgradient(model) do m
            masked_mlm_loss(m, x_batch, y_batch, n_classes)[1]
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, l_val)
        global global_step += 1
        if use_max_steps && global_step >= config["max_steps"]
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval (every epoch — used for checkpointing and sweep selection)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    for start_idx in 1:config["batch_size"]:size(X_val_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_val_masked, 2))
        x_batch = CuArray(X_val_masked[:, start_idx:end_idx])
        y_batch = CuArray(y_val_masked[:, start_idx:end_idx])
        loss_val, _, _ = masked_mlm_loss(model, x_batch, y_batch, n_classes)
        push!(val_eval_losses, loss_val)
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only — held out for reporting)
    is_last = is_last || done
    eval_losses = Float32[]
    epoch_rank_errors = Int[]
    epoch_preds = Int[]
    epoch_trues = Int[]

    if is_last
        for start_idx in 1:config["batch_size"]:size(X_test_masked, 2)
            end_idx = min(start_idx + config["batch_size"] - 1, size(X_test_masked, 2))

            x_batch = CuArray(X_test_masked[:, start_idx:end_idx])
            y_batch = CuArray(y_test_masked[:, start_idx:end_idx])

            loss_val, logits_masked, y_targets = masked_mlm_loss(model, x_batch, y_batch, n_classes)
            push!(eval_losses, loss_val)

            if !isnothing(y_targets)
                logits_cpu = cpu(logits_masked)
                y_targets_cpu = cpu(y_targets)

                append!(epoch_preds, Flux.onecold(logits_cpu))
                append!(epoch_trues, y_targets_cpu)

                for i in eachindex(y_targets_cpu)
                    r = y_targets_cpu[i]  # ETF: r = rank (target class in inverse_ranks mode); RTF: r = gene_id
                    col = @view logits_cpu[:, i]
                    err = count(x -> x > col[r], col)
                    push!(epoch_rank_errors, err)
                    if use_exp
                        rank_error_sums[r] += err
                        rank_error_counts[r] += 1
                    else
                        gene_error_sums[r] += err
                        gene_error_counts[r] += 1
                    end
                end

                y_labels_cpu = cpu(y_batch)
                batch_len = size(y_labels_cpu, 2)
                masked_idx = 0
                for j in 1:batch_len
                    for pos in 1:n_genes  # ETF: pos = gene position; RTF: pos = rank position
                        r = y_labels_cpu[pos, j]
                        (r == -100 || r <= 0 || r > n_classes) && continue
                        masked_idx += 1
                        col = @view logits_cpu[:, masked_idx]
                        err = count(x -> x > col[r], col)
                        if use_exp
                            gene_error_sums[pos] += err
                            gene_error_counts[pos] += 1
                        else
                            rank_error_sums[pos] += err
                            rank_error_counts[pos] += 1
                        end
                    end
                end
            end
        end
    end
    if !isempty(eval_losses)
        push!(test_losses, mean(eval_losses))
        push!(test_rank_errors, isempty(epoch_rank_errors) ? NaN32 : mean(Float32.(epoch_rank_errors)))
    end

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

    if is_last
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "logit-ce")
cs, cp = plot_ranked_heatmap(all_trues, all_preds, save_dir)
plot_per_rank_error(rank_error_sums, rank_error_counts, n_genes, save_dir)
plot_per_gene_error(gene_error_sums, gene_error_counts, n_genes, save_dir,
                    "mean rank error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, n_genes, save_dir,
                           "mean rank error", "per_rank_error")

# log_model(model, save_dir)
log_model(model, save_dir, config)
log_info(; save_dir=save_dir, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test_masked=X_test_masked, y_test_masked=y_test_masked,
           X_test=use_exp ? X_expr[:, test_indices] : X_test)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, pearson=cp, spearman=cs,
           total_steps=global_step, best_epoch=best_epoch,
           best_val_loss=best_val_loss)

wb !== nothing && wandb.finish()

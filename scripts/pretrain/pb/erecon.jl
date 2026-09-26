using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics, Functors
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
using Preprocess, Models, Train, Log, Plot, Args, Config, ProcessLabels

args = load_pretrain_args()
config = load_config(args["config"], args;
                     hp_section=["pretrain", "erecon", args["modeltype"]])

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

# gene selection: ETF = top-n HVGs, RTF = full gene vocab + per-sample top_k truncation (matches mlm + finetune)
n_hvg = get(config, "n_hvg", 0)
if use_exp
    if n_hvg > 0 && n_hvg < size(data_expr, 1)
        n_orig = size(data_expr, 1)
        data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
        println("HVG filter: $(n_orig) → $(n_hvg) genes")
    end
else
    println("RTF: using full gene vocab ($(size(data_expr, 1)) genes), top_k truncation")
end

# train-split medians from file (scripts/pretrain/pb/compute_medians.jl), HVG subset for etf
gene_medians = gene_medians_for(config, data_expr; hvg_idx=(@isdefined(hvg_idx) ? hvg_idx : nothing))
X_ranks = rank_genes(data_expr, gene_medians)
X_expr = data_expr isa Matrix{Float32} ? data_expr : Float32.(data_expr) # avoid a full copy when already Float32

n_genes = size(X_ranks, 1)
MASK_ID = n_genes + 1
top_k = get(config, "top_k", 1024)
if !use_exp && top_k < n_genes
    X_ranks = X_ranks[1:top_k, :]
    println("top_k truncation: $(n_genes) → $top_k ranked genes per sample")
end
seq_len = use_exp ? n_genes : min(top_k, n_genes)

# splits
# seeded so every PB pretrain objective/modeltype shares one split; reseed after so masks/donors stay random per run
Random.seed!(get(config, "split_seed", 42))
_, _, _, train_indices, val_indices, test_indices = tvsplit(X_ranks, 0.1f0, 0.1f0)
Random.seed!()

if use_exp
    X_train = X_expr[:, train_indices]
    X_val = X_expr[:, val_indices]
    X_test = X_expr[:, test_indices]
else
    X_train = X_ranks[:, train_indices]
    X_val = X_ranks[:, val_indices]
    X_test = X_ranks[:, test_indices]
end
# ETF: X_* already are the expression splits (no second copy); RTF: full expression needed for label lookup
X_expr_train, X_expr_val, X_expr_test = use_exp ? (X_train, X_val, X_test) :
    (X_expr[:, train_indices], X_expr[:, val_indices], X_expr[:, test_indices])
X_ranks_test = use_exp ? X_ranks[:, test_indices] : X_test
if use_exp
    inv_ranks_test = inverse_ranks(X_ranks_test)
end

# model
model = if use_exp
    ExpEReconModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                   n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                   dropout_prob=config["drop_prob"])
else
    RankEReconModel(n_genes=n_genes, embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                    dropout_prob=config["drop_prob"], seq_len=seq_len)
end
model = cu(model)
model = fix_gpu_dropout(model)
opt = Flux.setup(OptimiserChain(ClipNorm(1.0), AdamW(config["lr"])), model)

# mask val
X_val_masked = similar(X_val)
if use_exp
    val_corrupt_mask = falses(size(X_val))
    corrupt_expr!(X_val_masked, val_corrupt_mask, X_val, config["mask_ratio"])
else
    y_val_labels = fill(-100f0, size(X_val))
    mask_input_erecon!(X_val_masked, y_val_labels, X_val, X_expr_val, config["mask_ratio"], -100f0, MASK_ID)
end

# mask test
X_test_masked = similar(X_test)
if use_exp
    test_corrupt_mask = falses(size(X_test))
    corrupt_expr!(X_test_masked, test_corrupt_mask, X_test, config["mask_ratio"])
else
    y_test_labels = fill(-100f0, size(X_test))
    mask_input_erecon!(X_test_masked, y_test_labels, X_test, X_expr_test, config["mask_ratio"], -100f0, MASK_ID)
end

# save dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
dataset_tag = fmt == "lincs" ? joinpath("lincs") : joinpath("tahoe", "pb")
save_dir = joinpath("results", dataset_tag, "pretrain", "erecon", config["modeltype"], timestamp)
mkpath(save_dir)
println("save dir: $save_dir")
# save split now (survives a crash before log_info) and next to every checkpoint (finetune reads <model_dir>/indices.jld2)
save_indices(dir) = jldsave(joinpath(dir, "indices.jld2");
                            train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)
save_indices(save_dir)
if @isdefined(hvg_idx) # not defined when HVG filter skipped (RTF, or LINCS 978 < n_hvg)
    jldsave(joinpath(save_dir, "hvg_indices.jld2"); hvg_idx=hvg_idx)
end

wandb = init_wandb(config, "PB-PT-Aug", "erecon_$(fmt)_$(config["modeltype"])_$(timestamp)")
wb = config["wandb_mode"] != "disabled" ? wandb : nothing

# train
train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]
gene_error_sums = zeros(Float32, n_genes)
gene_error_counts = zeros(Int, n_genes)
rank_error_sums = zeros(Float32, n_genes)
rank_error_counts = zeros(Int, n_genes)

global_step = 0
use_max_steps = config["max_steps"] > 0
done = false
best_val_loss = Inf32
best_epoch = 0

bpe = cld(size(X_train, 2), config["batch_size"])
n_total_epochs = use_max_steps ? cld(config["max_steps"], bpe) : config["n_epochs"]
# per-step lr schedule (matches SC pretrain); per-epoch version ran the whole last epoch at lr = 0
total_steps = use_max_steps ? config["max_steps"] : n_total_epochs * bpe
warmup_steps = max(1, div(total_steps, 10))
train_loss_maxes = Float32[]
X_train_masked = similar(X_train)
if use_exp
    train_corrupt_mask = falses(size(X_train))
else
    y_train_labels = fill(-100f0, size(X_train))
end

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)

    if use_exp
        corrupt_expr!(X_train_masked, train_corrupt_mask, X_train, config["mask_ratio"])
    else
        mask_input_erecon!(X_train_masked, y_train_labels, X_train, X_expr_train, config["mask_ratio"], -100f0, MASK_ID)
    end

    # train epoch
    Flux.trainmode!(model)
    epoch_losses = Float32[]

    for start_idx in 1:config["batch_size"]:size(X_train_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_train_masked, 2))

        x_batch = CuArray(X_train_masked[:, start_idx:end_idx])
        Optimisers.adjust!(opt, compute_lr_step(global_step + 1, total_steps, config["lr"], warmup_steps))

        if use_exp
            y_batch = CuArray(X_expr_train[:, start_idx:end_idx])
            m_batch = CuArray(Float32.(train_corrupt_mask[:, start_idx:end_idx]))

            l_val, grads = Flux.withgradient(model) do m
                masked_erecon_loss(m, x_batch, y_batch, m_batch)[1]
            end
        else
            y_batch = CuArray(y_train_labels[:, start_idx:end_idx])

            l_val, grads = Flux.withgradient(model) do m
                masked_erecon_loss(m, x_batch, y_batch)[1]
            end
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, l_val)
        global global_step += 1
        if use_max_steps && global_step >= config["max_steps"]
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))
    push!(train_loss_maxes, maximum(epoch_losses))

    # val eval (every epoch for checkpt + sweep selection)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    for start_idx in 1:config["batch_size"]:size(X_val_masked, 2)
        end_idx = min(start_idx + config["batch_size"] - 1, size(X_val_masked, 2))
        x_batch = CuArray(X_val_masked[:, start_idx:end_idx])
        if use_exp
            y_batch = CuArray(X_expr_val[:, start_idx:end_idx])
            m_batch = CuArray(Float32.(val_corrupt_mask[:, start_idx:end_idx]))
            loss_val, _, _ = masked_erecon_loss(model, x_batch, y_batch, m_batch)
        else
            y_batch = CuArray(y_val_labels[:, start_idx:end_idx])
            loss_val, _, _ = masked_erecon_loss(model, x_batch, y_batch)
        end
        push!(val_eval_losses, cpu(loss_val))
    end
    push!(val_losses, mean(val_eval_losses))

    # save best before the last-epoch reload so a best final epoch isn't overwritten by the older checkpoint
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        mkpath(joinpath(save_dir, "best"))
        log_model(model, joinpath(save_dir, "best"), config)
        isfile(joinpath(save_dir, "best", "indices.jld2")) || save_indices(joinpath(save_dir, "best"))
    end

    # test eval (final epoch only)
    is_last = is_last || done

    # reload best checkpoint for test eval
    if is_last && isfile(joinpath(save_dir, "best", "model_state.jld2"))
        best_state = load(joinpath(save_dir, "best", "model_state.jld2"))["model_state"]
        model_cpu = cpu(model)
        Flux.loadmodel!(model_cpu, best_state)
        global model = fix_gpu_dropout(cu(model_cpu))
        println("reloaded best model (epoch $best_epoch) for test eval")
    end

    eval_losses = Float32[]
    epoch_preds = Float32[]
    epoch_trues = Float32[]

    if is_last
        for start_idx in 1:config["batch_size"]:size(X_test_masked, 2)
            end_idx = min(start_idx + config["batch_size"] - 1, size(X_test_masked, 2))

            x_batch = CuArray(X_test_masked[:, start_idx:end_idx])

            if use_exp
                y_batch = CuArray(X_expr_test[:, start_idx:end_idx])
                m_batch = CuArray(Float32.(test_corrupt_mask[:, start_idx:end_idx]))

                loss_val, preds_masked, y_masked = masked_erecon_loss(model, x_batch, y_batch, m_batch)
                push!(eval_losses, cpu(loss_val))

                if !isnothing(preds_masked)
                    preds_cpu = cpu(preds_masked)
                    y_cpu = cpu(y_masked)
                    append!(epoch_preds, preds_cpu)
                    append!(epoch_trues, y_cpu)

                    m_cpu = test_corrupt_mask[:, start_idx:end_idx]
                    inv_ranks_batch = inv_ranks_test[:, start_idx:end_idx]
                    batch_len = size(m_cpu, 2)
                    masked_idx = 0
                    for j in 1:batch_len
                        for pos in 1:n_genes # ETF: pos = gene index
                            m_cpu[pos, j] || continue
                            masked_idx += 1
                            sq_err = (preds_cpu[masked_idx] - y_cpu[masked_idx])^2
                            gene_error_sums[pos] += sq_err # per-gene
                            gene_error_counts[pos] += 1
                            r = inv_ranks_batch[pos, j] # rank of gene pos
                            rank_error_sums[r] += sq_err # per-rank
                            rank_error_counts[r] += 1
                        end
                    end
                end
            else
                y_batch = CuArray(y_test_labels[:, start_idx:end_idx])

                loss_val, preds_masked, y_masked = masked_erecon_loss(model, x_batch, y_batch)
                push!(eval_losses, cpu(loss_val))

                if !isnothing(preds_masked)
                    preds_cpu = cpu(preds_masked)
                    y_cpu_masked = cpu(y_masked)
                    append!(epoch_preds, preds_cpu)
                    append!(epoch_trues, y_cpu_masked)

                    y_labels_cpu = cpu(y_batch)
                    ranks_batch = X_ranks_test[:, start_idx:end_idx]
                    batch_len = size(y_labels_cpu, 2)
                    masked_idx = 0
                    for j in 1:batch_len
                        for pos in 1:size(y_labels_cpu, 1) # RTF: pos = rank position (top_k rows)
                            (y_labels_cpu[pos, j] == -100f0) && continue
                            masked_idx += 1
                            sq_err = (preds_cpu[masked_idx] - y_cpu_masked[masked_idx])^2
                            rank_error_sums[pos] += sq_err # per-rank
                            rank_error_counts[pos] += 1
                            gene_id = ranks_batch[pos, j] # gene at rank pos
                            gene_error_sums[gene_id] += sq_err # per-gene
                            gene_error_counts[gene_id] += 1
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
                        "val_loss" => val_losses[end], "train_loss_max" => train_loss_maxes[end],
                        "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end

    # moved above the reload

    if is_last
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "MSE";
         val_losses=val_losses)

cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

fig_scatter = Figure(size=(600, 400))
ax_scatter = Axis(fig_scatter[1, 1], xlabel="true expression", ylabel="predicted expression")
n_plot = min(length(all_trues), 50000)
plot_idx = sample(1:length(all_trues), n_plot, replace=false)
scatter!(ax_scatter, all_trues[plot_idx], all_preds[plot_idx], markersize=1, alpha=0.3)
text!(ax_scatter, 0.05, 0.95, space=:relative, align=(:left, :top),
    text="Pearson: $(round(cp, digits=4))\nSpearman: $(round(cs, digits=4))")
save(joinpath(save_dir, "scatter.png"), fig_scatter)

plot_per_gene_error(gene_error_sums, gene_error_counts, n_genes, save_dir,
                    "mean squared error", "per_gene_error";
                    sorted_gene_path=get(config, "sorted_gene_path", ""))
plot_per_sample_rank_error(rank_error_sums, rank_error_counts, n_genes, save_dir,
                           "mean squared error", "per_rank_error")

log_model(model, save_dir, config)
log_info(; save_dir=save_dir, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test_masked=X_test_masked,
           train_loss_maxes=train_loss_maxes,
           y_test_masked=use_exp ? X_expr_test : y_test_labels)
           # X_test=use_exp ? X_expr[:, test_indices] : X_test)  # ignored by log_info

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=pretrain_skip, pearson=cp, spearman=cs,
           total_steps=global_step, best_epoch=best_epoch,
           best_val_loss=best_val_loss)

wb !== nothing && wandb.finish()

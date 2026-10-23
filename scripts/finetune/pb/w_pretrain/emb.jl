using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, Extract, FTModels

args = load_finetune_args()
# config = load_config(args["config"], args)

# emb mode tag: emb (mlp probe) or emb_linear
# emb_tag = something(args["probe"], "mlp") == "linear" ? "emb_linear" : "emb"
# + _raw for unstandardized embeddings (--emb_std 0)
emb_std = something(args["emb_std"], 1) == 1
emb_tag = (something(args["probe"], "mlp") == "linear" ? "emb_linear" : "emb") * (emb_std ? "" : "_raw")
config = load_config(args["config"], args,
                     hp_section=["finetune", "w_pretrain", args["modeltype"], wpt_task_tag(args), args["level"], emb_tag])
resolve_data_path!(config)
resolve_model_dir!(config)
Extract.FINAL_LN[] = something(get(config, "final_ln", nothing), 0) == 1
resolve_lvl3_cells!(config)

# seed
seed = get(config, "seed", nothing)
if !isnothing(seed)
    Random.seed!(seed)
    CUDA.seed!(seed)
    println("Random seed: $seed")
end

is_regression = config["level"] == "lvl3"
use_oversmpl = config["level"] == "lvl2" && !is_regression

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

start_time = now()
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")

# data
fmt = get(config, "data_format", "tahoe")
data_key = fmt == "lincs" ? "filtered_data" : "df"
data = load(config["data_path"])[data_key]
if fmt == "lincs"
    data_expr = data isa Matrix{Float32} ? data : Float32.(data.expr)
    meta_df = data.inst
else
    data_expr = Float32.(reduce(hcat, data.expr))
    meta_df = data
end


# gene selection
n_genes_orig = size(data_expr, 1)
if config["modeltype"] == "etf"
    n_hvg = get(config, "n_hvg", 0)
    if n_hvg > 0 && n_hvg < size(data_expr, 1)
        data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
        println("HVG filter: $(n_genes_orig) → $(n_hvg) genes")
    end
end

# result folder tag: <mt>, + _gdd / _gpl for the group split
model_tag = config["modeltype"]
something(get(config, "group_split", nothing), 0) == 1 && (model_tag *= fmt == "lincs" ? "_gpl" : "_gdd")
string(something(get(config, "input", nothing), "abs")) == "delta" && (model_tag *= "_delta")

d = dsplit(data_expr, config;
           label_path=get(config, "label_path", ""),
           label_source=(fmt == "tahoe" ? meta_df : nothing),
           inst_df=(fmt == "lincs" && !isa(data, Matrix) ? data.inst : nothing),
           gene_df=(fmt == "lincs" && !isa(data, Matrix) ? data.gene : nothing),
           ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes)

# lvl3 identity baseline
id_baseline = is_regression ? d.id_baseline : nothing

if config["modeltype"] == "rtf"
    # RTF: top-k
    top_k = get(config, "top_k", 1024)
    if top_k < d.n_genes
        # d = merge(d, (X_train = d.X_train[1:top_k, :],
                       # X_val   = d.X_val[1:top_k, :],
                       # X_test  = d.X_test[1:top_k, :]))

        d = merge(d, (X_train = d.X_train[1:top_k, :],
                       X_val   = d.X_val[1:top_k, :],
                       X_test  = d.X_test[1:top_k, :],
                       X_test_full = (hasproperty(d, :X_test_full) ? d.X_test_full[1:top_k, :] : nothing)))
        println("top_k truncation: $(d.n_genes) → $top_k ranked genes per sample")
    end
    # seq_len = top_k
    # n_genes_for_model = d.n_genes

    seq_len = min(top_k, d.n_genes)
    n_genes_for_model = d.n_genes
else
    # ETF: HVG applied above
    seq_len = d.n_genes
    n_genes_for_model = n_genes_orig
end

# embedding-only model
# ft_model, train_input, val_input, test_input = build_embm(config, d.X_train, d.X_test,
                                                # n_genes_for_model, d.n_classifications; X_val=d.X_val,
                                                # seq_len=seq_len)
# opt = Flux.setup(Optimisers.AdamW(config["lr"]), ft_model)

# grouped split: embed the full test groups once, test = its pretrain-split subset
has_full = hasproperty(d, :test_full_idx) && d.test_full_idx !== d.test_idx
X_test_emb = has_full ? d.X_test_full : d.X_test
# ft_model, train_input, val_input, test_input_full = build_embm(config, d.X_train, X_test_emb,
                                                # n_genes_for_model, d.n_classifications; X_val=d.X_val,
                                                # seq_len=seq_len, hvg_idx=(@isdefined(hvg_idx) ? hvg_idx : nothing))
# frozen embeddings cached per encoder x split (sweep runs reuse them); cache stores the split indices and is checked on load
test_emb_idx = has_full ? d.test_full_idx : d.test_idx
# emb_key = join([realpath(config["model_dir"]), config["level"], model_tag, seq_len, Extract.FINAL_LN[],
                # size(d.X_train, 2), size(d.X_val, 2), size(X_test_emb, 2)], "|")
emb_key = join([realpath(config["model_dir"]), config["level"], model_tag, seq_len, Extract.FINAL_LN[],
                size(d.X_train, 2), size(d.X_val, 2), size(X_test_emb, 2), wpt_run_tag(config)], "|")
emb_cache = joinpath("results", fmt == "lincs" ? "lincs" : joinpath("tahoe", "pb"), "finetune", "w_pretrain", "emb_cache",
                     "$(config["modeltype"])_$(wpt_run_tag(config))_$(config["level"])_$(string(hash(emb_key), base=16)).jld2")
cached = isfile(emb_cache) ? load(emb_cache) : nothing
if !isnothing(cached) && cached["key"] == emb_key && cached["train_idx"] == d.train_idx && cached["val_idx"] == d.val_idx &&
   cached["test_idx"] == test_emb_idx
    train_input, val_input, test_input_full = cached["train"], cached["val"], cached["test"]
    # ft_model = fix_gpu_dropout(cu(probe_head(config, size(train_input, 1), d.n_classifications)))
    println("embeddings: loaded cache $emb_cache")
else
    ft_model, train_input, val_input, test_input_full = build_embm(config, d.X_train, X_test_emb,
                                                    n_genes_for_model, d.n_classifications; X_val=d.X_val,
                                                    seq_len=seq_len, hvg_idx=(@isdefined(hvg_idx) ? hvg_idx : nothing))
    mkpath(dirname(emb_cache))
    tmp_cache = "$(emb_cache).$(getpid()).tmp"
    jldsave(tmp_cache; key=emb_key, train_idx=d.train_idx, val_idx=d.val_idx, test_idx=test_emb_idx,
            train=train_input, val=val_input, test=test_input_full)
    mv(tmp_cache, emb_cache; force=true)
    println("embeddings: wrote cache $emb_cache")
end
# probe head rebuilt after a reseed, so cached and uncached runs start from the same weights
isnothing(seed) || Random.seed!(seed)
ft_model = fix_gpu_dropout(cu(probe_head(config, size(train_input, 1), d.n_classifications)))
test_input = has_full ? test_input_full[:, something.(indexin(d.test_idx, d.test_full_idx))] : test_input_full
# both probes: z-score embeddings with train mean/sd (embedding scale differs ~300x across encoders)
# emb_stats = standardize_embeds!(train_input, val_input, test_input, has_full ? test_input_full : nothing)
# --emb_std 0: raw embeddings (test_input is a copy, so test_input_full is z-scored separately)
emb_stats = emb_std ? standardize_embeds!(train_input, val_input, test_input, has_full ? test_input_full : nothing) : nothing
wd = Float64(something(get(config, "weight_decay", nothing), 0.0))
opt = Flux.setup(Optimisers.AdamW(config["lr"], (0.9, 0.999), wd), ft_model)

# save dir
dataset_tag = fmt == "lincs" ? "lincs" : joinpath("tahoe", "pb")
# save_dir = joinpath("results", dataset_tag, "finetune", "w_pretrain", config["level"],
                    # config["modeltype"], config["task"], "emb", timestamp)
# mkpath(save_dir)

save_dir = joinpath("results", dataset_tag, "finetune", "w_pretrain", config["level"], model_tag,
                    wpt_run_tag(config), emb_tag, isnothing(seed) ? timestamp : "$(timestamp)_s$(seed)")
mkpath(save_dir)
isnothing(emb_stats) || jldsave(joinpath(save_dir, "emb_stats.jld2"); mu=emb_stats[1], sd=emb_stats[2])
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
# wandb = init_wandb(config, wandb_project(config, "FT"), "emb_$(fmt)_$(config["modeltype"])_$(config["level"])$(seed_tag)_$(timestamp)")

wandb = init_wandb(config, wandb_project(config, "FT"), "$(config["modeltype"])_$(wpt_run_tag(config))_$(emb_tag)_$(fmt)_$(config["level"])$(seed_tag)_$(timestamp)")
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
# --patience: stop after this many evals without val improvement (0 = off)
patience = something(get(config, "patience", nothing), 0)
since_best = 0
best_val_loss = Inf32
best_val_acc = -Inf  # classification: best/ = highest val accuracy (val loss bottoms out early under held-out wells)
val_accs = Float64[]
best_epoch = 0
n_total_epochs = if use_max_steps
    bpe = div(size(train_input, 2), config["batch_size"])
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

# test eval for model m
# function run_test(m)
    # epoch_preds = is_regression ? Float32[] : Int[]
    # epoch_trues = is_regression ? Float32[] : Int[]
    # eval_losses = Float32[]
    # n_test = size(test_input, 2)
    # for s in 1:config["batch_size"]:n_test
        # e = min(s + config["batch_size"] - 1, n_test)
        # x_gpu = cu(test_input[:, s:e])
        # y_gpu = cu(d.y_test[:, s:e])

function run_test(m; X_test=test_input, y_test=d.y_test)
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]
    eval_losses = Float32[]
    n_test = size(X_test, 2)
    for s in 1:config["batch_size"]:n_test
        e = min(s + config["batch_size"] - 1, n_test)
        x_gpu = cu(X_test[:, s:e])
        y_gpu = cu(y_test[:, s:e])
        logits = m(x_gpu)
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
    return mean(eval_losses), epoch_preds, epoch_trues
end

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)

    # train epoch
    Flux.trainmode!(ft_model)
    epoch_losses = Float32[]
    n_train = size(train_input, 2)
    num_batches = div(n_train, config["batch_size"])
    perm = use_oversmpl ? nothing : randperm(n_train)

    for i in 1:num_batches
        if use_oversmpl
            batch_idx = [rand(d.cidx_dict[rand(d.cs)]) for _ in 1:config["batch_size"]]
        else
            s = (i - 1) * config["batch_size"] + 1
            e = min(s + config["batch_size"] - 1, n_train)
            batch_idx = perm[s:e]
        end

        x_gpu = cu(train_input[:, batch_idx])
        y_gpu = cu(d.y_train[:, batch_idx])

        lv, grads = Flux.withgradient(ft_model) do m
            preds = m(x_gpu)
            is_regression ? Flux.mse(preds, y_gpu) : Flux.logitcrossentropy(preds, y_gpu)
        end
        Flux.update!(opt, ft_model, grads[1])
        push!(epoch_losses, Float32(cpu(lv)))
        global global_step += 1
        if use_max_steps && global_step >= ft_step_limit
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval
    Flux.testmode!(ft_model)
    val_eval_losses = Float32[]
    val_correct = 0; val_n = 0
    n_val = size(val_input, 2)
    for s in 1:config["batch_size"]:n_val
        e = min(s + config["batch_size"] - 1, n_val)
        x_gpu = cu(val_input[:, s:e])
        y_gpu = cu(d.y_val[:, s:e])
        logits = ft_model(x_gpu)
        if is_regression
            push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
        else
            push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            val_correct += sum(Flux.onecold(cpu(logits)) .== Flux.onecold(cpu(y_gpu))); val_n += size(y_gpu, 2)
        end
    end
    push!(val_losses, mean(val_eval_losses))
    if patience > 0 && !is_regression
        va = val_correct / max(val_n, 1)
        if va > best_val_acc || (va == best_val_acc && val_losses[end] < best_val_loss)
            global since_best = 0
        else
            global since_best += 1
            since_best >= patience && (global done = true)
        end
    end

    # test eval (final epoch)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        final_test_loss, epoch_preds, epoch_trues = run_test(ft_model)
        push!(test_losses, final_test_loss)
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    push!(val_accs, is_regression ? NaN : val_correct / max(val_n, 1))
    # regression: lowest val loss; classification: highest val accuracy, ties -> lower val loss
    improved = is_regression ? val_losses[end] < best_val_loss :
               (val_accs[end] > best_val_acc || (val_accs[end] == best_val_acc && val_losses[end] < best_val_loss))
    if improved
        global best_val_loss = val_losses[end]
        global best_val_acc = val_accs[end]
        global best_epoch = epoch
        best_dir = joinpath(save_dir, "best")
        mkpath(best_dir)

        log_model(ft_model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, is_regression ? "MSE" : "CE")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)

        if is_regression
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_skip,
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_skip, total_steps=global_step,
                best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        end
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                         # "val_loss" => val_losses[end], "val_acc" => val_accs[end], "global_step" => global_step)

                         "val_loss" => val_losses[end], "val_acc" => val_accs[end], "global_step" => global_step,
                         "best_val_loss" => best_val_loss,  # per-epoch so hyperband early_terminate can see it
                         "best_val_acc" => best_val_acc)    # sweep metric (classification): matches best/ selection
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end
end


# best-model test eval
opt = nothing; GC.gc(true); CUDA.reclaim()  # free optimizer state
best_cpu = load_best_cpu(ft_model, save_dir)
best_preds, best_trues = if isnothing(best_cpu)
    println("no best/ checkpoint found, best metrics = final model")
    all_preds, all_trues
else
    best_model = fix_gpu_dropout(cu(best_cpu))
    Flux.testmode!(best_model)
    _, bp, bt = run_test(best_model)
    bp, bt
end
best_metrics = test_metrics(best_preds, best_trues, is_regression)
# final_metrics = test_metrics(all_preds, all_trues, is_regression)
# isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues)

# grouped split: also test on the full test groups
best_full_metrics = if has_full
    m_full = isnothing(best_cpu) ? ft_model : fix_gpu_dropout(cu(best_cpu))
    Flux.testmode!(m_full)
    _, fp, ft = run_test(m_full; X_test=test_input_full, y_test=d.y_test_full)
    test_metrics(fp, ft, is_regression)
else
    nothing
end
has_full && println("test (best model, full test groups): ", best_full_metrics)
final_metrics = test_metrics(all_preds, all_trues, is_regression)
# with sample ids + class names
isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues,
    test_sample_idx=(hasproperty(d, :label_idx) ? collect(d.label_idx[d.test_idx]) : collect(d.test_idx)),
    label_names=(hasproperty(d, :label_names) ? d.label_names : String[]),
    split_tag=string(get(d, :split_tag, "lvl3")))
println("test (best model, epoch $best_epoch): ", best_metrics)
println("test (final model):         ", final_metrics)

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_val_acc"] = best_val_acc
    wb.summary["best_epoch"] = best_epoch
    for (k, v) in pairs(best_metrics); wb.summary["best_$(k)"] = v; end
    for (k, v) in pairs(final_metrics); wb.summary["final_$(k)"] = v; end
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

log_model(ft_model, save_dir)
log_info(; save_dir=save_dir, train_indices=d.train_idx, val_indices=d.val_idx, test_indices=d.test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=test_input)

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
               skip=finetune_skip, r2=r2, pearson=pearson, rmse=rmse,
               best_r2=best_metrics.r2, best_pearson=best_metrics.pearson, best_rmse=best_metrics.rmse,
               final_r2=final_metrics.r2, final_pearson=final_metrics.pearson, final_rmse=final_metrics.rmse,
               id_r2=isnothing(id_baseline) ? NaN : id_baseline.r2,
               id_pearson=isnothing(id_baseline) ? NaN : id_baseline.pearson,
               id_rmse=isnothing(id_baseline) ? NaN : id_baseline.rmse,
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               # skip=finetune_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               # best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)

               skip=finetune_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               split_tag=string(get(d, :split_tag, "")), best_accuracy_fullgroup=(isnothing(best_full_metrics) ? NaN : best_full_metrics.accuracy),
               best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
end

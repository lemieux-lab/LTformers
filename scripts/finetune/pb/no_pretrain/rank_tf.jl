using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, FTModels

args = load_finetune_args()
config = load_config(args["config"], args)
resolve_data_path!(config)
resolve_model_dir!(config)

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
else  # tahoe
    data_expr = Float32.(reduce(hcat, data.expr))
    meta_df = data
end

n_hvg = get(config, "n_hvg", 0)
if n_hvg > 0 && n_hvg < size(data_expr, 1)
    n_orig = size(data_expr, 1)
    data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
    println("HVG filter: $(n_orig) → $(n_hvg) genes")
end

d = dsplit(data_expr, config;
           label_path=get(config, "label_path", ""),
           label_source=(fmt == "tahoe" ? meta_df : nothing),
           inst_df=(fmt == "lincs" && !isa(data, Matrix) ? data.inst : nothing),
           gene_df=(fmt == "lincs" && !isa(data, Matrix) ? data.gene : nothing),
           ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes)

n_genes = d.n_genes
n_classifications = d.n_classifications
X_train, X_val, X_test = d.X_train, d.X_val, d.X_test
y_train, y_val, y_test = d.y_train, d.y_val, d.y_test
train_idx, val_idx, test_idx = d.train_idx, d.val_idx, d.test_idx
cidx_dict, cs = d.cidx_dict, d.cs

# model
model = RankClassifier(n_genes=n_genes, embed_dim=config["embed_dim"],
                       n_layers=config["n_layers"], n_classifications=n_classifications,
                       n_heads=config["n_heads"], hidden_dim=config["hidden_dim"],
                       dropout_prob=config["drop_prob"])
model = fix_gpu_dropout(cu(model))
# opt = Flux.setup(Optimisers.Adam(config["lr"]), model)
opt = Flux.setup(Optimisers.AdamW(config["lr"]), model)

# save dir
dataset_tag = fmt == "lincs" ? "lincs" : joinpath("tahoe", "pb")
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], "rtf", timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
wandb = init_wandb(config, "PB-FT-Aug", "rtf_nopt_$(fmt)_$(config["level"])$(seed_tag)_$(timestamp)")
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
# best_test_loss = Inf32
best_val_loss = Inf32
best_epoch = 0
n_total_epochs = if use_max_steps
    bpe = div(size(X_train, 2), config["batch_size"])
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

for epoch in ProgressBar(1:n_total_epochs)
    done && break
    is_last = (epoch == n_total_epochs)

    # train epoch
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    n_train = size(X_train, 2)
    num_batches = div(n_train, config["batch_size"])
    perm = use_oversmpl ? nothing : randperm(n_train)

    for i in 1:num_batches
        if use_oversmpl
            batch_idx = [rand(cidx_dict[rand(cs)]) for _ in 1:config["batch_size"]]
        else
            s = (i - 1) * config["batch_size"] + 1
            e = min(s + config["batch_size"] - 1, n_train)
            batch_idx = perm[s:e]
        end

        x_gpu = cu(Int32.(X_train[:, batch_idx]))
        y_gpu = cu(y_train[:, batch_idx])

        lv, grads = Flux.withgradient(model) do m
            preds = m(x_gpu)
            is_regression ? Flux.mse(preds, y_gpu) : Flux.logitcrossentropy(preds, y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, Float32(cpu(lv)))
        global global_step += 1
        if use_max_steps && global_step >= ft_step_limit
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval (every epoch — used for checkpoint selection)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    n_val = size(X_val, 2)
    for s in 1:config["batch_size"]:n_val
        e = min(s + config["batch_size"] - 1, n_val)
        x_gpu = cu(Int32.(X_val[:, s:e]))
        y_gpu = cu(y_val[:, s:e])
        logits = model(x_gpu)
        if is_regression
            push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
        else
            push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
        end
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only — held out for reporting)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        eval_losses = Float32[]
        n_test = size(X_test, 2)
        for s in 1:config["batch_size"]:n_test
            e = min(s + config["batch_size"] - 1, n_test)
            x_gpu = cu(Int32.(X_test[:, s:e]))
            y_gpu = cu(y_test[:, s:e])
            logits = model(x_gpu)
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
        push!(test_losses, mean(eval_losses))
        append!(all_preds, epoch_preds)
        append!(all_trues, epoch_trues)
    end

    # if test_losses[end] < best_test_loss
    #     global best_test_loss = test_losses[end]
    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        best_dir = joinpath(save_dir, "best")
        mkpath(best_dir)

        log_model(model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, is_regression ? "MSE" : "logit-ce")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)

        if is_regression
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_no_pt_skip,
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=finetune_no_pt_skip, total_steps=global_step,
                best_epoch=best_epoch, best_val_loss=best_val_loss)
        end
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                         "val_loss" => val_losses[end], "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end
end

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_epoch"] = best_epoch
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "logit-ce")

log_model(model, save_dir)
log_info(; save_dir=save_dir, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=X_test)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

if is_regression
    r2 = 1.0 - sum((all_preds .- all_trues) .^ 2) / sum((all_trues .- mean(all_trues)) .^ 2)
    pearson = cor(all_preds, all_trues)
    rmse = sqrt(mean((all_preds .- all_trues) .^ 2))
    println("R² = $(round(r2, digits=4)), Pearson r = $(round(pearson, digits=4)), RMSE = $(round(rmse, digits=4))")
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=finetune_no_pt_skip, r2=r2, pearson=pearson, rmse=rmse,
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=finetune_no_pt_skip, accuracy=acc, total_steps=global_step,
               best_epoch=best_epoch, best_val_loss=best_val_loss)
end

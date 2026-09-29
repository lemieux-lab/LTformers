using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, FTModels

args = load_finetune_args()
config = load_config(args["config"], args,
                     hp_section=["finetune", "no_pretrain", args["modeltype"], args["level"]])
resolve_data_path!(config)
resolve_model_dir!(config)
resolve_lvl3_cells!(config)
config["modeltype"] == "elog" || error("elog.jl requires -t elog (got $(config["modeltype"]))")

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

n_hvg = get(config, "n_hvg", 0)
# lvl3: HVGs applied after pairing
n_genes_all = size(data_expr, 1)
hvg_idx_lvl3 = nothing
if n_hvg > 0 && n_hvg < n_genes_all
    if config["level"] == "lvl3"
        _, hvg_idx_lvl3 = select_hvg(data_expr, n_hvg)
        println("HVG filter (lvl3, applied after pairing): $(n_genes_all) → $(n_hvg) genes")
    else
        data_expr, hvg_idx = select_hvg(data_expr, n_hvg)
        println("HVG filter: $(n_genes_all) → $(n_hvg) genes")
    end
end
n_used = (n_hvg > 0 && n_hvg < n_genes_all) ? n_hvg : n_genes_all
model_tag = gene_set_tag(config["modeltype"], n_used, n_genes_all; kind="hvg")
something(get(config, "group_split", nothing), 0) == 1 && (model_tag *= "_gdd")  # (drug, dose) group split
println("gene set: $n_used of $n_genes_all genes → saving as $model_tag")

d = dsplit(data_expr, config;
           label_path=get(config, "label_path", ""),
           label_source=(fmt == "tahoe" ? meta_df : nothing),
           inst_df=(fmt == "lincs" && !isa(data, Matrix) ? data.inst : nothing),
           gene_df=(fmt == "lincs" && !isa(data, Matrix) ? data.gene : nothing),
           ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes,
           hvg_idx_lvl3=hvg_idx_lvl3)

# lvl3 identity baseline
id_baseline = is_regression ? d.id_baseline : nothing

# per-gene z-score with train-split mean/sd, then clamp to [-clip, clip] (Scanpy/Seurat: 10). default on;
# --standardize 0 = raw inputs, --clip 0 = no clipping. raw log-expression is badly conditioned (uncentered offsets,
# per-gene sd spread up to ~3500x): elog/emlp +10-15 pts on LINCS and TPB HVG with it
if something(get(config, "standardize", nothing), 1) == 1
    μ = mean(d.X_train, dims=2)
    σ = std(d.X_train, dims=2)
    σ[σ .< 1f-6] .= 1f0                      # constant genes -> centered only
    clip_c = Float32(something(get(config, "clip", nothing), 10))
    z(X) = (X .- μ) ./ σ
    Z_train = z(d.X_train)
    if clip_c > 0
        println("clip ±$clip_c: $(round(100 * mean(abs.(Z_train) .> clip_c), digits=4))% of train values clipped")
        zc(X) = clamp.(z(X), -clip_c, clip_c)
        d = merge(d, (; X_train=clamp.(Z_train, -clip_c, clip_c), X_val=zc(d.X_val), X_test=zc(d.X_test)))
        model_tag *= "_zc"
    else
        d = merge(d, (; X_train=Z_train, X_val=z(d.X_val), X_test=z(d.X_test)))
        model_tag *= "_z"
    end
    Z_train = nothing
    println("standardized inputs with train mean/sd → saving as $model_tag")
end

# model
# linear layer: logreg (lvl1/2) / linreg (lvl3)
config["lr"] = 0.001
model = Flux.Chain(Flux.Dense(d.n_genes => d.n_classifications))
model = cu(model)
opt = Flux.setup(Optimisers.AdamW(config["lr"]), model)

# save dir
dataset_tag = fmt == "lincs" ? "lincs" : joinpath("tahoe", "pb")
seed_tag = isnothing(seed) ? "" : "_s$(seed)"
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], model_tag, "$(timestamp)$(seed_tag)")
mkpath(save_dir)
println("save dir: $save_dir")

wandb = init_wandb(config, wandb_project(config, "npt-FT"), "$(model_tag)_nopt_$(fmt)_$(config["level"])$(seed_tag)_$(timestamp)")
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
best_val_loss = Inf32
best_val_acc = -Inf  # classification: best/ = highest val accuracy (val loss bottoms out early under held-out wells)
val_accs = Float64[]
best_epoch = 0
n_total_epochs = if use_max_steps
    bpe = div(size(d.X_train, 2), config["batch_size"])
    cld(ft_step_limit, max(bpe, 1))
else
    config["n_epochs"]
end

# test eval for model m
function run_test(m)
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]
    eval_losses = Float32[]
    n_test = size(d.X_test, 2)
    for s in 1:config["batch_size"]:n_test
        e = min(s + config["batch_size"] - 1, n_test)
        x_gpu = cu(d.X_test[:, s:e])
        y_gpu = cu(d.y_test[:, s:e])
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
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    n_train = size(d.X_train, 2)
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

        x_gpu = cu(d.X_train[:, batch_idx])
        y_gpu = cu(d.y_train[:, batch_idx])

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

    # val eval
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    val_correct = 0; val_n = 0
    n_val = size(d.X_val, 2)
    for s in 1:config["batch_size"]:n_val
        e = min(s + config["batch_size"] - 1, n_val)
        x_gpu = cu(d.X_val[:, s:e])
        y_gpu = cu(d.y_val[:, s:e])
        logits = model(x_gpu)
        if is_regression
            push!(val_eval_losses, Float32(cpu(Flux.mse(logits, y_gpu))))
        else
            push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            val_correct += sum(Flux.onecold(cpu(logits)) .== Flux.onecold(cpu(y_gpu))); val_n += size(y_gpu, 2)
        end
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch)
    is_last = is_last || done
    epoch_preds = is_regression ? Float32[] : Int[]
    epoch_trues = is_regression ? Float32[] : Int[]

    if is_last
        final_test_loss, epoch_preds, epoch_trues = run_test(model)
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
        log_model(model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, is_regression ? "MSE" : "CE")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)
        if is_regression
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=mlp_skip,
                total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        else
            log_params(config, gpu_info, 0, 0, best_dir;
                skip=mlp_skip, total_steps=global_step,
                best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        end
    end

    if wb !== nothing
        log_dict = Dict("epoch" => epoch, "train_loss" => train_losses[end],
                         "val_loss" => val_losses[end], "val_acc" => val_accs[end], "global_step" => global_step)
        if !isempty(test_losses)
            log_dict["test_loss"] = test_losses[end]
        end
        wb.log(log_dict)
    end
end


# best-model test eval
opt = nothing; GC.gc(true); CUDA.reclaim()  # free optimizer state
best_cpu = load_best_cpu(model, save_dir)
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
final_metrics = test_metrics(all_preds, all_trues, is_regression)
isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=best_preds, all_trues=best_trues)
println("test (best model, epoch $best_epoch): ", best_metrics)
println("test (final model):         ", final_metrics)

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_epoch"] = best_epoch
    for (k, v) in pairs(best_metrics); wb.summary["best_$(k)"] = v; end
    for (k, v) in pairs(final_metrics); wb.summary["final_$(k)"] = v; end
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

log_model(model, save_dir)
log_info(; save_dir=save_dir, train_indices=d.train_idx, val_indices=d.val_idx, test_indices=d.test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=d.X_test)

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
               skip=mlp_skip, r2=r2, pearson=pearson, rmse=rmse,
               best_r2=best_metrics.r2, best_pearson=best_metrics.pearson, best_rmse=best_metrics.rmse,
               final_r2=final_metrics.r2, final_pearson=final_metrics.pearson, final_rmse=final_metrics.rmse,
               id_r2=isnothing(id_baseline) ? NaN : id_baseline.r2,
               id_pearson=isnothing(id_baseline) ? NaN : id_baseline.pearson,
               id_rmse=isnothing(id_baseline) ? NaN : id_baseline.rmse,
               total_steps=global_step, best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=mlp_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
end

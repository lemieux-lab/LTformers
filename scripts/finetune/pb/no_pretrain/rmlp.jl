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
# all-gene runs (--rank_top_k 0, tahoe only; lincs has 978 genes): [finetune.no_pretrain.<model>.<lvl>.full] overrides the 1024 hps
if get(config, "data_format", "tahoe") != "lincs" && something(get(config, "rank_top_k", nothing), 1024) == 0
    Config._merge_hp!(config, ["finetune", "no_pretrain", config["modeltype"], config["level"], "full"], args)
end
resolve_data_path!(config)
resolve_model_dir!(config)
resolve_lvl3_cells!(config)
config["modeltype"] == "rmlp" || error("rmlp.jl is MLP only; use rlog.jl for -t rlog (got $(config["modeltype"]))")

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
# job id + pid: parallel sweep agents / jobs starting in the same minute otherwise share a save_dir (NFS stale file handle)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", "0") * "_p" * string(getpid())

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

# no HVG, rank features

d = dsplit(data_expr, config;
           label_path=get(config, "label_path", ""),
           label_source=(fmt == "tahoe" ? meta_df : nothing),
           inst_df=(fmt == "lincs" && !isa(data, Matrix) ? data.inst : nothing),
           gene_df=(fmt == "lincs" && !isa(data, Matrix) ? data.gene : nothing),
           ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes,
           inverse_ranks_fn=inverse_ranks)

rank_k = rank_feature_k(config, d.n_genes)
model_tag = gene_set_tag(config["modeltype"], rank_k, d.n_genes; kind="topk")
something(get(config, "group_split", nothing), 0) == 1 && (model_tag *= fmt == "lincs" ? "_gpl" : "_gdd")  # (drug, dose) group split
println("rank features: k=$rank_k of $(d.n_genes) genes → saving as $model_tag")

# lvl3 identity baseline
id_baseline = is_regression ? d.id_baseline : nothing

# model
# tapered relu MLP (mlp_hidden_dim = 0), else n_layers constant-width hidden layers
mlp_h = something(get(config, "mlp_hidden_dim", nothing), 0)
mlp_shape = something(get(config, "mlp_shape", nothing), "const")  # const | funnel (halve width each layer, floor n_classes)
mlp_hidden = mlp_shape == "funnel" ? [max(mlp_h ÷ 2^(i-1), d.n_classifications) for i in 1:config["n_layers"]] :
                                     fill(mlp_h, config["n_layers"])
sizes = mlp_h > 0 ? [d.n_genes; mlp_hidden; d.n_classifications] :
        [round(Int, d.n_genes + (d.n_classifications - d.n_genes) * i / (config["n_layers"] + 1))
         for i in 0:config["n_layers"]+1]
println("MLP sizes: $sizes")
layers = []
for i in 1:length(sizes)-1
    push!(layers, Flux.Dense(sizes[i] => sizes[i+1], i < length(sizes)-1 ? relu : identity))
    if i < length(sizes) - 1
        push!(layers, Flux.Dropout(config["drop_prob"]))
    end
end
model = Flux.Chain(layers...)
model = fix_gpu_dropout(cu(model))
opt = Flux.setup(Optimisers.AdamW(config["lr"]), model)

# save dir
dataset_tag = fmt == "lincs" ? "lincs" : joinpath("tahoe", "pb")
seed_tag = isnothing(seed) ? "" : "_s$(seed)"  # in dir name: seeded runs launched in the same minute would share it
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], model_tag, "$(timestamp)$(seed_tag)")
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
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
best_state = nothing  # best-val weights, kept on cpu in RAM; written to disk once after training
save_model = get(config, "save_model", 1) != 0  # 0: no model_state.jld2 files (sweeps; full-gene MLPs are 1-4 GB each)
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
        x_gpu = cu(Float32.(d.X_test[:, s:e]))
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

        x_gpu = cu(Float32.(d.X_train[:, batch_idx]))
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
        x_gpu = cu(Float32.(d.X_val[:, s:e]))
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
        # log_model(model, best_dir)
        global best_state = Flux.state(cpu(model))
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
# best_cpu = load_best_cpu(model, save_dir)
if save_model && !isnothing(best_state)
    jldsave(joinpath(save_dir, "best", "model_state.jld2"); model_state=best_state)
end
best_cpu = isnothing(best_state) ? nothing : Flux.loadmodel!(cpu(model), best_state)
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
    wb.summary["best_val_acc"] = best_val_acc
    wb.summary["best_epoch"] = best_epoch
    for (k, v) in pairs(best_metrics); wb.summary["best_$(k)"] = v; end
    for (k, v) in pairs(final_metrics); wb.summary["final_$(k)"] = v; end
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

# log_model(model, save_dir)
save_model && log_model(model, save_dir)
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

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, FTModels
using LoadSC, ProcessSC

args = load_sc_finetune_args()
config = load_config(args["config"], args)
config["data_format"] = "tahoe_sc"
# resolve_model_dir!(config)  # no pretrain weights needed

# seed
seed = get(config, "seed", nothing)
if !isnothing(seed)
    Random.seed!(seed)
    CUDA.seed!(seed)
    println("Random seed: $seed")
end

is_regression = false  # SC finetune: lvl1/lvl2 classification only
use_oversmpl = config["level"] == "lvl2"

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

start_time = now()
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")

# data — SC shard loading
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])

# HVG loading (MLP uses expression features like ETF)
hvg_idx = nothing
hvg_path = get(config, "hvg_path", "")
if hvg_path != "" && isfile(hvg_path)
    hvg_data = JLD2.load(hvg_path)
    hvg_idx = hvg_data["hvg_idx"]
    println("loaded $(length(hvg_idx)) HVG indices")
end

top_k = get(config, "top_k", 1024)
d = load_sc_finetune_data(all_shards, config["level"], token_to_idx, n_coding, top_k, "etf";
                           pb_data_path=get(config, "pb_data_path", ""),
                           hvg_idx=hvg_idx,
                           subset_shards=get(config, "subset_shards", 0),
                           process_cell_topk_flat_fn=process_cell_topk_flat,
                           cell_to_dense_flat_fn=cell_to_dense_flat!,
                           oversmpl_fn=oversmpl)

# model — MLP with linearly interpolated layer sizes
sizes = [round(Int, d.n_genes + (d.n_classifications - d.n_genes) * i / (config["n_layers"] + 1))
         for i in 0:config["n_layers"]+1]
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
dataset_tag = joinpath("tahoe", "sc")
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], "mlp", timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
wandb = init_wandb(config, "SC-FT-Aug", "mlp_nopt_sc_$(config["level"])$(seed_tag)_$(timestamp)")
wb = get(config, "wandb_mode", "disabled") != "disabled" ? wandb : nothing

# train
train_losses = Float32[]
val_losses = Float32[]
test_losses = Float32[]
all_preds = Int[]
all_trues = Int[]

global_step = 0
ft_step_limit = get(config, "max_ft_steps", 0)
use_max_steps = ft_step_limit > 0
done = false
best_val_loss = Inf32
best_epoch = 0
n_total_epochs = if use_max_steps
    bpe = div(size(d.X_train, 2), config["batch_size"])
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
            Flux.logitcrossentropy(preds, y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, Float32(cpu(lv)))
        global global_step += 1
        if use_max_steps && global_step >= ft_step_limit
            global done = true; break
        end
    end
    push!(train_losses, mean(epoch_losses))

    # val eval (every epoch for checkpt selection)
    Flux.testmode!(model)
    val_eval_losses = Float32[]
    n_val = size(d.X_val, 2)
    for s in 1:config["batch_size"]:n_val
        e = min(s + config["batch_size"] - 1, n_val)
        x_gpu = cu(d.X_val[:, s:e])
        y_gpu = cu(d.y_val[:, s:e])
        logits = model(x_gpu)
        push!(val_eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
    end
    push!(val_losses, mean(val_eval_losses))

    # test eval (final epoch only)
    is_last = is_last || done

    if is_last
        eval_losses = Float32[]
        n_test = size(d.X_test, 2)
        for s in 1:config["batch_size"]:n_test
            e = min(s + config["batch_size"] - 1, n_test)
            x_gpu = cu(d.X_test[:, s:e])
            y_gpu = cu(d.y_test[:, s:e])
            logits = model(x_gpu)
            push!(eval_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))
            append!(all_preds, Flux.onecold(cpu(logits)))
            append!(all_trues, Flux.onecold(cpu(y_gpu)))
        end
        push!(test_losses, mean(eval_losses))
    end

    if val_losses[end] < best_val_loss
        global best_val_loss = val_losses[end]
        global best_epoch = epoch
        best_dir = joinpath(save_dir, "best")
        mkpath(best_dir)
        log_model(model, best_dir)
        plot_loss(length(train_losses), train_losses, val_losses, best_dir, "CE")
        jldsave(joinpath(best_dir, "losses.jld2"); epochs=1:epoch,
                train_losses=train_losses, val_losses=val_losses)
        log_params(config, gpu_info, 0, 0, best_dir;
            skip=mlp_skip, total_steps=global_step,
            best_epoch=best_epoch, best_val_loss=best_val_loss)
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
plot_loss(length(train_losses), train_losses, test_losses, save_dir, "CE")

log_model(model, save_dir)
log_info(; save_dir=save_dir, train_indices=d.train_idx, val_indices=d.val_idx, test_indices=d.test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=d.X_test)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)
run_hours, run_minutes = div(total_minutes, 60), rem(total_minutes, 60)

acc = mean(all_preds .== all_trues)
log_params(config, gpu_info, run_hours, run_minutes, save_dir;
           skip=mlp_skip, accuracy=acc, total_steps=global_step,
           best_epoch=best_epoch, best_val_loss=best_val_loss)

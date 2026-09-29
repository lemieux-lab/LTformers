using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../../..", arch_dir)))

using JLD2, CUDA, Dates, Flux, Optimisers, Random, Statistics
using ProgressBars, CairoMakie, StatsBase, DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../../src/tahoe"))
using Models, Train, Log, Plot, Args, Config, ProcessLabels, Preprocess, FTModels
using LoadSC, ProcessSC, SCTrain

args = load_sc_finetune_args()
config = load_config(args["config"], args,
                     hp_section=["finetune", "no_pretrain", args["modeltype"], args["level"]],
                     dataset="tahoe_sc")
config["data_format"] = "tahoe_sc"
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

CUDA.device!(0)
gpu_info = CUDA.name(device())
println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "N/A"))

start_time = now()
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM") * "_j" * get(ENV, "SLURM_JOB_ID", string(getpid()))
# --sc_split: "drug_dose" (default; all wells of a drug-dose held out together), "well" (whole wells),
# or "well_cl" ((well, cell line) units, like a random PB split)
sc_split = something(get(config, "sc_split", nothing), "drug_dose")
config["sc_split"] = sc_split  # logged to params.txt (default changed well -> drug_dose on 2026-09-28)
sc_split == "well_cl" && (timestamp *= "_wcl")
sc_split == "well" && (timestamp *= "_well")

# data
coding_tokens, token_to_idx, n_coding = load_gene_vocab(config["meta_dir"], config["coding_gene_path"])
all_shards = list_shards(config["data_dir"])

# rank features, --rank_top_k 0 = all genes
top_k = rank_feature_k(config, n_coding)
model_tag = gene_set_tag(config["modeltype"], top_k, n_coding; kind="topk")
println("rank features: top-$top_k of $n_coding genes → saving as $model_tag")
# PB data for per-cell lvl3
sc_lvl3_percell = !get(config, "sc_lvl3_pseudobulk", false)
pb_expr_for_percell = nothing
pb_df_for_percell = nothing
if sc_lvl3_percell && config["level"] == "lvl3"
    pb_path = get(config, "pb_data_path", "")
    pb_path == "" && error("sc_lvl3_percell requires pb_data_path in config")
    println("loading PB data for per-cell SC lvl3 targets: $pb_path")
    pb_data = load(pb_path)["df"]
    pb_expr_for_percell = Float32.(reduce(hcat, pb_data.expr))
    pb_df_for_percell = pb_data
end

d = load_sc_finetune_data_streaming(all_shards, config["level"], token_to_idx, n_coding, top_k, "rtf";
                           split_by=sc_split,
                           pb_data_path=get(config, "pb_data_path", ""),
                           subset_shards=get(config, "subset_shards", 0),
                           process_cell_topk_flat_fn=process_cell_topk_flat,
                           cell_to_dense_flat_fn=cell_to_dense_flat!,
                           oversmpl_fn=oversmpl,
                           source_cell=get(config, "source_cell", ""),
                           target_cell=get(config, "target_cell", ""),
                           dose=get(config, "dose", ""),
                           meta_dir=get(config, "meta_dir", ""),
                           regression_pairs_fn=get_regression_pairs_pca,
                           sc_lvl3_percell=sc_lvl3_percell,
                           pb_expr=pb_expr_for_percell,
                           pb_df=pb_df_for_percell,
                           actual_modeltype=config["modeltype"],
                           identity_baseline_fn=identity_baseline)
is_streaming = d.train_shard_map !== nothing

# val: fixed subset of ft_eval_shards shards (default 100); test: all (ft_test_shards)
val_paths, test_paths = sc_eval_paths(d, config)

id_baseline = is_regression ? d.id_baseline : nothing

# RTF gene ids -> rank features
function sc_inverse_ranks(X_rtf::Matrix{Int32}, n_coding::Int)
    inv = zeros(Float32, n_coding, size(X_rtf, 2))
    for j in axes(X_rtf, 2)
        for r in axes(X_rtf, 1)
            g = X_rtf[r, j]
            if g > 0 && g <= n_coding
                inv[g, j] = Float32(r)
            end
        end
    end
    return inv
end

if is_streaming
    println("streaming mode: all inverse ranks (train/val/test) computed per-batch")
    X_train = nothing
elseif eltype(d.X_train) == Float32
    # lvl3 per-cell: already encoded
    println("lvl3 mode: rank features already encoded by the loader")
    X_train, X_val, X_test = d.X_train, d.X_val, d.X_test
else
    # gene-id tokens -> rank features
    println("converting top-$(size(d.X_train, 1)) gene ids to rank features (n_coding=$n_coding)...")
    _feat(ids) = rank_features(ids, vec(sum(ids .!= pad_token_id(n_coding), dims=1)), n_coding, size(ids, 1))
    X_train, X_val, X_test = _feat(d.X_train), _feat(d.X_val), _feat(d.X_test)
end
n_genes = n_coding
n_classifications = d.n_classifications
if !is_streaming
    y_val = d.y_val
    y_test = d.y_test
    y_train = d.y_train
end
train_idx, val_idx, test_idx = d.train_idx, d.val_idx, d.test_idx
println("inverse ranks done: input dim = $n_genes")

# model
# tapered relu MLP (mlp_hidden_dim = 0), else n_layers constant-width hidden layers
# sizes = [round(Int, n_genes + (n_classifications - n_genes) * i / (config["n_layers"] + 1))
#          for i in 0:config["n_layers"]+1]
mlp_h = something(get(config, "mlp_hidden_dim", nothing), 0)
mlp_shape = something(get(config, "mlp_shape", nothing), "const")  # const | funnel (halve width each layer, floor n_classes)
mlp_hidden = mlp_shape == "funnel" ? [max(mlp_h ÷ 2^(i-1), n_classifications) for i in 1:config["n_layers"]] :
                                     fill(mlp_h, config["n_layers"])
# sizes = mlp_h > 0 ? [n_genes; fill(mlp_h, config["n_layers"]); n_classifications] :
sizes = mlp_h > 0 ? [n_genes; mlp_hidden; n_classifications] :
        [round(Int, n_genes + (n_classifications - n_genes) * i / (config["n_layers"] + 1))
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
dataset_tag = joinpath("tahoe", "sc")
save_dir = joinpath("results", dataset_tag, "finetune", "no_pretrain", config["level"], model_tag, timestamp)
mkpath(save_dir)
println("save dir: $save_dir")

seed_tag = isnothing(seed) ? "" : "_s$(seed)"
wandb = init_wandb(config, wandb_project(config, "npt-FT"; sc=true), "$(model_tag)_nopt_sc_$(config["level"])$(seed_tag)_$(timestamp)")
wb = get(config, "wandb_mode", "disabled") != "disabled" ? wandb : nothing

# train: cross-shard mixed batches (class-balanced for lvl2), val every ft_eval_every steps, best-val checkpoint
fk = (; token_to_idx, n_coding, top_k, batch_size=config["batch_size"], feat_mt="rankfeat",
        n_cls=d.n_classifications, hvg_idx=nothing, to_x=x -> CuArray(x),
        process_cell_topk_flat_fn=process_cell_topk_flat, cell_to_dense_flat_fn=cell_to_dense_flat!)
r = sc_finetune!(model, opt, is_streaming ? d : merge(d, (; X_train, X_val, X_test)), fk, config; is_regression=is_regression, save_dir=save_dir,
                 gpu_info=gpu_info, skip=mlp_skip, loss_name=is_regression ? "MSE" : "CE", wb=wb,
                 val_paths=val_paths, test_paths=test_paths)
train_losses, val_losses, test_losses = r.train_losses, r.val_losses, r.test_losses
all_preds, all_trues = r.all_preds, r.all_trues
best_step, best_val_loss, best_val_acc, global_step = r.best_step, r.best_val_loss, r.best_val_acc, r.global_step

opt = nothing; GC.gc(true); CUDA.reclaim()  # free optimizer state
best_metrics = test_metrics(r.best_preds, r.best_trues, is_regression)
final_metrics = test_metrics(all_preds, all_trues, is_regression)
isdir(joinpath(save_dir, "best")) && jldsave(joinpath(save_dir, "best", "predstrues.jld2"); all_preds=r.best_preds, all_trues=r.best_trues)
println("test (best model, step $best_step): ", best_metrics)
println("test (final model, step $global_step): ", final_metrics)

if wb !== nothing
    wb.summary["best_val_loss"] = best_val_loss
    wb.summary["best_step"] = best_step
    for (k, v) in pairs(best_metrics); wb.summary["best_$(k)"] = v; end
    for (k, v) in pairs(final_metrics); wb.summary["final_$(k)"] = v; end
    wandb.finish()
end

# log
plot_loss(length(train_losses), train_losses, test_losses, save_dir, is_regression ? "MSE" : "CE")

log_model(model, save_dir)
log_info(; save_dir=save_dir, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
           n_epochs=length(train_losses), train_losses=train_losses,
           val_losses=val_losses, test_losses=test_losses,
           all_preds=all_preds, all_trues=all_trues,
           X_test=nothing)

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
               total_steps=global_step, best_step=best_step, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
else
    acc = mean(all_preds .== all_trues)
    log_params(config, gpu_info, run_hours, run_minutes, save_dir;
               skip=mlp_skip, accuracy=acc, best_accuracy=best_metrics.accuracy, final_accuracy=final_metrics.accuracy, total_steps=global_step,
               best_well_acc=r.best_agg.well_acc, best_wellcl_acc=r.best_agg.wellcl_acc,
               final_well_acc=r.final_agg.well_acc, final_wellcl_acc=r.final_agg.wellcl_acc,
               best_step=best_step, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
end

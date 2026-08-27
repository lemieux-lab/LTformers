module Log

using Flux, JLD2

export pretrain_skip, finetune_skip, finetune_no_pt_skip, mlp_skip
export log_model, log_info, log_params


# skip sets

const pretrain_skip = Set([
    "model_dir", "mode", "task", "level", "max_ft_steps", "wandb_mode"])

const finetune_skip = Set([
    "model_dir", "mask_ratio", "subset_ratio", "max_steps", "ema_decay"])

const finetune_no_pt_skip = Set([
    "model_dir", "mode", "task", "mask_ratio", "subset_ratio", "max_steps", "ema_decay"])

const mlp_skip = Set([
    "model_dir", "mode", "task", "mask_ratio", "subset_ratio", "max_steps", "ema_decay",
    "embed_dim", "hidden_dim", "n_heads", "n_layers"])


# logging

function log_model(model, save_dir::String)
    model_cpu = cpu(model)
    jldsave("$save_dir/model_state.jld2"; model_state=Flux.state(model_cpu))
end

# overload: also save architecture config alongside model state
function log_model(model, save_dir::String, config::Dict)
    log_model(model, save_dir)
    arch_keys = ["embed_dim", "hidden_dim", "n_heads", "n_layers", "drop_prob", "modeltype"]
    arch_config = Dict{String,Any}(k => config[k] for k in arch_keys if haskey(config, k))
    jldsave("$save_dir/model_config.jld2"; arch_config=arch_config)
end

function log_info(; train_indices,
                    test_indices,
                    val_indices = nothing,
                    n_epochs::Int,
                    train_losses,
                    test_losses,
                    val_losses = nothing,
                    save_dir::String,
                    all_preds = nothing,
                    all_trues = nothing,
                    target_variances = nothing,
                    X_test_masked = nothing,
                    y_test_masked = nothing,
                    X_test = nothing)

    idx_kwargs = Dict{Symbol,Any}(:train_indices => train_indices,
                                  :test_indices => test_indices)
    if !isnothing(val_indices)
        idx_kwargs[:val_indices] = val_indices
    end
    jldsave(joinpath(save_dir, "indices.jld2"); idx_kwargs...)

    loss_kwargs = Dict{Symbol,Any}(:epochs => 1:n_epochs,
                                   :train_losses => train_losses,
                                   :test_losses => test_losses)
    if !isnothing(val_losses)
        loss_kwargs[:val_losses] = val_losses
    end
    if !isnothing(target_variances)
        loss_kwargs[:target_variances] = target_variances
    end
    jldsave(joinpath(save_dir, "losses.jld2"); loss_kwargs...)

    if !isnothing(all_preds)
        jldsave(joinpath(save_dir, "predstrues.jld2");
                all_preds = all_preds, all_trues = all_trues)
    end
    if !isnothing(X_test_masked)
        jldsave(joinpath(save_dir, "masked_test_data.jld2");
                X = X_test_masked, y = y_test_masked)
    end
    if !isnothing(X_test)
        jldsave(joinpath(save_dir, "test_data.jld2"); X = X_test)
    end
end

## old log_params -- alphabetical order
# function log_params(config::Dict, gpu_info::String, run_hours, run_minutes, save_dir::String;
#                     skip::Set{String} = Set{String}(), metrics...)
#     open(joinpath(save_dir, "params.txt"), "w") do io
#         println(io, "PARAMETERS:\n###########\n$(gpu_info)")
#         # for (k, v) in sort(collect(config))
#         for (k, v) in sort(collect(config), by=first)
#             k in skip && continue
#             println(io, "$k = $v")
#         end
#         println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
#         if !isempty(metrics)
#             println(io, "\nMETRICS:\n###########")
#             for (k, v) in metrics
#                 println(io, "$k = $v")
#             end
#         end
#     end
# end

const _param_groups = [
    "# data paths" => ["data_path", "data_format", "modeltype", "sorted_gene_path"],
    "# single-cell data paths" => ["data_dir", "meta_dir", "coding_gene_path"],
    "# parameters" => [
        "batch_size", "n_epochs", "embed_dim", "hidden_dim", "n_heads", "n_layers",
        "n_hvg", "top_k", "n_eval_shards", "subset_shards", "subset_ratio", "max_steps",
        "lr", "drop_prob", "mask_ratio", "ema_decay",
    ],
    "# finetune" => ["model_dir", "mode", "task", "level", "max_ft_steps", "label_path"],
    "# lvl3 references/targets" => ["source_cell", "target_cell", "target_gene"],
    "# misc" => ["wandb_mode", "additional_notes"],
]

function log_params(config::Dict, gpu_info::String, run_hours, run_minutes, save_dir::String;
                    skip::Set{String} = Set{String}(), metrics...)
    open(joinpath(save_dir, "params.txt"), "w") do io
        println(io, "$(gpu_info)\n")
        logged = Set{String}()
        for (header, keys) in _param_groups
            printed_header = false
            for k in keys
                k in skip && continue
                !haskey(config, k) && continue
                if !printed_header
                    println(io, header)
                    printed_header = true
                end
                println(io, "$k = $(config[k])")
                push!(logged, k)
            end
            printed_header && println(io)
        end
        # any remaining config keys not in the groups above
        remaining = sort(collect(k for k in keys(config) if !(k in logged) && !(k in skip)))
        if !isempty(remaining)
            println(io, "# other")
            for k in remaining
                println(io, "$k = $(config[k])")
            end
            println(io)
        end
        println(io, "# runtime")
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
        if !isempty(metrics)
            println(io, "\n# metrics")
            for (k, v) in metrics
                println(io, "$k = $v")
            end
        end
    end
end


end  # module Log

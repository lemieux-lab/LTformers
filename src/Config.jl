module Config

using TOML, PyCall

export load_config, gpu_lr, init_wandb, resolve_model_dir!, resolve_data_path!


function _merge_local!(config::Dict, toml_path::String)
    local_path = replace(toml_path, r"\.toml$" => ".local.toml")
    if !isfile(local_path)
        # also check for local.toml in the same directory
        local_path = joinpath(dirname(toml_path), "local.toml")
    end
    if isfile(local_path)
        local_config = TOML.parsefile(local_path)
        for (k, v) in local_config
            config[k] = v
        end
        println("merged local config: $local_path")
    end
    return config
end

function load_config(toml_path::String; overrides...)
    config = TOML.parsefile(toml_path)
    _merge_local!(config, toml_path)
    for (k, v) in overrides
        config[String(k)] = v
    end
    return config
end

function load_config(toml_path::String, args::Dict{String, Any})
    config = TOML.parsefile(toml_path)
    _merge_local!(config, toml_path)
    for (k, v) in args
        !isnothing(v) && (config[k] = v)
    end
    return config
end

function gpu_lr(base_lr::Float64, batch_size::Int; base_batch::Int = 128)
    return base_lr * (batch_size / base_batch)
end

function init_wandb(config::Dict, project::String, run_name::String)
    if get(config, "wandb_mode", "disabled") == "disabled"
        return nothing
    end
    wandb = pyimport("wandb")
    wandb.init(project=project, config=config, mode=config["wandb_mode"], name=run_name)
    return wandb
end


function resolve_model_dir!(config::Dict;
                           models_toml::String = joinpath(@__DIR__, "../config/pretrained_models.toml"))
    model_dir = get(config, "model_dir", "")
    if model_dir != ""
        return config
    end

    fmt = get(config, "data_format", "tahoe")
    task = get(config, "task", "mlm")
    modeltype = get(config, "modeltype", "rtf")

    dataset_key = if fmt == "lincs"
        "lincs"
    elseif fmt == "tahoe"
        "tahoe_pb"
    elseif fmt == "tahoe_sc"
        "tahoe_sc"
    else
        error("resolve_model_dir!: unknown data_format '$fmt' (expected lincs, tahoe, or tahoe_sc)")
    end

    models = TOML.parsefile(models_toml)
    if !haskey(models, dataset_key) || !haskey(models[dataset_key], task)
        error("resolve_model_dir!: no entry for [$dataset_key.$task] in $models_toml")
    end
    section = models[dataset_key][task]
    if !haskey(section, modeltype)
        fallback = nothing
        for fb in ("rtf", "etf")
            if haskey(section, fb) && section[fb] != ""
                fallback = fb
                break
            end
        end
        if isnothing(fallback)
            error("resolve_model_dir!: no '$modeltype' key in [$dataset_key.$task] in $models_toml")
        end
        println("resolve_model_dir!: no '$modeltype' key, falling back to '$fallback' for indices")
        modeltype = fallback
    end

    dir = section[modeltype]
    if dir == ""
        error("resolve_model_dir!: model_dir for $dataset_key.$task.$modeltype is empty in $models_toml")
    end

    if !isabspath(dir)
        repo_root = abspath(joinpath(@__DIR__, ".."))
        dir = joinpath(repo_root, dir)
    end
    config["model_dir"] = dir
    println("resolved model_dir: $dir")
    return config
end


const DATA_PATHS = Dict(
    "lincs" => "data/lincs/lincs_trt_data.jld2",
    "tahoe" => "data/tahoe/filtered_pseudobulks_alpha_10000.jld2",
)

const LABEL_PATHS = Dict(
    "lincs" => "data/lincs/lincs_trt_inst.jld2",
)

function resolve_data_path!(config::Dict)
    fmt = get(config, "data_format", "tahoe")
    repo_root = abspath(joinpath(@__DIR__, ".."))

    # if data_path was already set (e.g. from local.toml or CLI), use it
    existing = get(config, "data_path", "")
    if existing != "" && isfile(existing)
        println("resolved data_path: $existing (from config)")
        path = existing
    else
        if !haskey(DATA_PATHS, fmt)
            error("resolve_data_path!: unknown data_format '$fmt' (expected lincs or tahoe)")
        end
        # repo_root = abspath(joinpath(@__DIR__, ".."))
        path = DATA_PATHS[fmt]
        if !isabspath(path)
            path = joinpath(repo_root, path)
        end
        config["data_path"] = path
        println("resolved data_path: $path")
    end

    if haskey(LABEL_PATHS, fmt) && get(config, "label_path", "") == ""
        lpath = LABEL_PATHS[fmt]
        if !isabspath(lpath)
            lpath = joinpath(repo_root, lpath)
        end
        config["label_path"] = lpath
        println("resolved label_path: $lpath")
    end

    return config
end


end  # module Config

module Args

using ArgParse

export load_pretrain_args, load_finetune_args, load_sc_finetune_args


function load_pretrain_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
        "--modeltype", "-t"
            help = "model type: rtf or etf"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--data_path"
            help = "path to JLD2 data file"
            arg_type = String
        "--data_format"
            help = "data format: tahoe or lincs"
            arg_type = String
        "--sorted_gene_path"
            help = "path to sorted_gene_indices_by_exp.jld2 for global-rank error plots"
            arg_type = String
        "--n_hvg"
            help = "number of highly variable genes to keep (0 = all)"
            arg_type = Int
        "--subset_ratio"
            help = "fraction of data to use (stratified by pert_type x cell line)"
            arg_type = Float64
        "--mask_ratio"
            help = "fraction of tokens to mask"
            arg_type = Float64
        "--max_steps"
            help = "max training steps (0 = use n_epochs)"
            arg_type = Int
        "--ema_decay"
            help = "EMA decay rate for teacher model"
            arg_type = Float64
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--data_dir"
            help = "path to Tahoe-100M parquet shard directory"
            arg_type = String
        "--meta_dir"
            help = "path to Tahoe-100M metadata directory"
            arg_type = String
        "--coding_gene_path"
            help = "path to protein-coding gene list TSV"
            arg_type = String
        "--top_k"
            help = "number of top genes per cell for sequence input"
            arg_type = Int
        "--n_eval_shards"
            help = "number of test shards to evaluate per epoch"
            arg_type = Int
        "--subset_shards"
            help = "max number of shards to use (0 = all); applied after train/test split"
            arg_type = Int
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
        "--hvg_n_shards"
            help = "number of shards to scan for HVG computation (compute_hvg.jl)"
            arg_type = Int
        "--hvg_out"
            help = "output path for HVG indices (compute_hvg.jl)"
            arg_type = String
    end
    return parse_args(s)
end

function load_finetune_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--mode", "-m"
            help = "ft mode: e2e or emb"
            arg_type = String
            required = true
        "--task"
            help = "pretrain objective: mlm, lrecon, or erecon"
            arg_type = String
        "--model_dir"
            help = "path to pretrained model directory (contains model_state.jld2)"
            arg_type = String
        "--level", "-l"
            help = "level of finetuning: lvl1, lvl2, or lvl3 (regression)"
            arg_type = String
            required = true
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
            required = true
        "--modeltype", "-t"
            help = "model type: rtf, emlp, rmlp, or etf"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--max_ft_steps"
            help = "max finetune steps (0 = use n_epochs)"
            arg_type = Int
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--data_path"
            help = "path to JLD2 data file"
            arg_type = String
        "--data_format"
            help = "data format: tahoe or lincs"
            arg_type = String
        "--label_path"
            help = "path to label JLD2 file (LINCS only; has mfc + pert_id)"
            arg_type = String
        "--n_hvg"
            help = "number of highly variable genes to keep (0 = all)"
            arg_type = Int
        "--subset_ratio"
            help = "fraction of data to use (stratified by pert_type x cell line)"
            arg_type = Float64
        "--sorted_gene_path"
            help = "path to sorted_gene_indices_by_exp.jld2 for global-rank error plots"
            arg_type = String
        "--source_cell"
            help = "source cell line for lvl3 regression (default: MCF7)"
            arg_type = String
        "--target_cell"
            help = "target cell line for lvl3 regression (default: PC3)"
            arg_type = String
        # "--target_gene"
        #     help = "target gene for lvl3 regression (default: IGFBP3)"
        #     arg_type = String
        "--dose"
            help = "dose filter for lvl3 regression (e.g. '5.0' for Tahoe, '10.0' for LINCS; empty = no filter)"
            arg_type = String
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
    end
    return parse_args(s)
end


function load_sc_finetune_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        # -- finetune args --
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--mode", "-m"
            help = "ft mode: e2e or emb"
            arg_type = String
            required = true
        "--task"
            help = "pretrain objective: mlm, lrecon, or erecon"
            arg_type = String
        "--model_dir"
            help = "path to pretrained model directory (contains model_state.jld2)"
            arg_type = String
        "--level", "-l"
            help = "level of finetuning: lvl1, lvl2, or lvl3 (regression)"
            arg_type = String
            required = true
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
            required = true
        "--modeltype", "-t"
            help = "model type: rtf, emlp, rmlp, or etf"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--max_ft_steps"
            help = "max finetune steps (0 = use n_epochs)"
            arg_type = Int
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
        # -- SC-specific args --
        "--data_dir"
            help = "path to Tahoe-100M parquet shard directory"
            arg_type = String
        "--meta_dir"
            help = "path to Tahoe-100M metadata directory (gene_vocabulary.jsonl)"
            arg_type = String
        "--coding_gene_path"
            help = "path to protein-coding gene list TSV"
            arg_type = String
        "--top_k"
            help = "number of top genes per cell for RTF sequence input"
            arg_type = Int
        "--n_hvg"
            help = "number of highly variable genes to keep (ETF; 0 = all)"
            arg_type = Int
        "--hvg_path"
            help = "path to pre-computed HVG indices JLD2 (for ETF)"
            arg_type = String
        "--subset_shards"
            help = "max number of shards to use (0 = all); for debugging"
            arg_type = Int
        "--pb_data_path"
            help = "path to PB JLD2 for determining valid lvl2 drugs"
            arg_type = String
        "--source_cell"
            help = "source cell line for lvl3 regression"
            arg_type = String
        "--target_cell"
            help = "target cell line for lvl3 regression"
            arg_type = String
        "--dose"
            help = "dose filter for lvl3 regression (e.g. '5.0'; empty = no filter)"
            arg_type = String
        # "--data_format"
        #     help = "data format (ignored for SC; kept for CLI compatibility with PB sweep launchers)"
        #     arg_type = String
        "--data_format"
            help = "data format (ignored for SC; kept for CLI compatibility with PB sweep launchers)"
            arg_type = String
    end
    return parse_args(s)
end


end  # module Args

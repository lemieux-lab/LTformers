using Pkg
arch = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
if Sys.ARCH == :aarch64
    Pkg.activate(joinpath(@__DIR__, "..", "..", "aarch64"))
else
    Pkg.activate(joinpath(@__DIR__, "..", "..", "x86_64"))
end
ENV["PYTHON"] = abspath("/home/golem/scratch/chans/transformers/.venv-$arch/bin/python")
if !haskey(Pkg.project().dependencies, "PyCall")
    Pkg.add("PyCall")
end
Pkg.build("PyCall")

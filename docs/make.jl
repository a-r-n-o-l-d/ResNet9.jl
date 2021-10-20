using ResNet9
using Documenter

DocMeta.setdocmeta!(ResNet9, :DocTestSetup, :(using ResNet9); recursive=true)

makedocs(;
    modules=[ResNet9],
    authors="Arnold",
    repo="https://github.com/a-r-n-o-l-d/ResNet9.jl/blob/{commit}{path}#{line}",
    sitename="ResNet9.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://a-r-n-o-l-d.github.io/ResNet9.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/a-r-n-o-l-d/ResNet9.jl",
    devbranch="main",
)

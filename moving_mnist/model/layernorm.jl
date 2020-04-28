mutable struct LayerNorm; a; b; ϵ; end

function LayerNorm(dmodel; eps=1e-5)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    return LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=2)
    #print("mean: ", size(μ))
    σ = std(x,mean=μ,dims=2)
    return l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b                                                         
end

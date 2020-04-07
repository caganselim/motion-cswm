struct LayerNorm; a; b; ϵ; end

function LayerNorm(dmodel; eps=1e-6)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    return LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1)
    return l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b                                                         
end

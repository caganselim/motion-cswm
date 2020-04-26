mutable struct EncoderMLP
    
    weights
    biases
    layer_norm
    act_fn

end

mutable struct LayerNorm2; a; b; ϵ; end

function LayerNorm2(dmodel; eps=1e-5)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    return LayerNorm(a, b, eps)
end

function (l::LayerNorm2)(x, o...)
    μ = mean(x,dims=3)
    #print("mean: ", size(μ))
    σ = std(x,mean=μ,dims=3)
    return l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b                                                         
end

function initEncoderMLP(input_dim, hidden_dim, output_dim, num_objects, act_fn)
    
    weights = [param(hidden_dim,input_dim),param(hidden_dim, hidden_dim),param(output_dim,hidden_dim )]
    biases =  [param0(hidden_dim), param0(hidden_dim), param0(output_dim)]
    layer_norm = LayerNorm2(hidden_dim)
    
    return EncoderMLP(weights, biases, layer_norm, act_fn)
    
end


function (e_mlp::EncoderMLP)(x)
    
    dims = size(x)
    x0 = reshape(x, dims[1]*dims[2], dims[3],dims[4])
    x0 = reshape(x0, dims[1]*dims[2],:)
    x1 = e_mlp.act_fn.(e_mlp.weights[1] * x0 .+ e_mlp.biases[1])
    x2 = e_mlp.weights[2]* x1 .+ e_mlp.biases[2]
    x3 = reshape(x2, :,dims[3],dims[4])
    x4 = e_mlp.layer_norm(x3)
    x5 = e_mlp.act_fn.(reshape(x4, :, dims[3]*dims[4]))
    x6 = e_mlp.weights[3]*x5 .+ e_mlp.biases[3]
    x7 = reshape(x6, 2,:,dims[4])
    
    return x7
    
end
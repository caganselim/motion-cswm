mutable struct EncoderCNNSmall
   
    weights
    biases
    bn_vars
    act_fn
    act_fn_hid
    
end

function initEncoderCNNSmall(input_dim, hidden_dim, num_objects, act_fn, act_fn_hid)
    
    weights = Any[param(10,10,3,hidden_dim), param(1,1,hidden_dim,num_objects)]
    biases = Any[param0(1,1,hidden_dim, 1), param0(1,1,num_objects,1)]
    bn_vars = Any[bnmoments(), atype(bnparams(hidden_dim))]
    
    return EncoderCNNSmall(weights, biases, bn_vars, act_fn, act_fn_hid)
    
end

function (e_cnn::EncoderCNNSmall)(x)
    
    #w(w_x,w_y,in_ch,out_ch)
    x1 = conv4(e_cnn.weights[1],x; stride = 10) .+ e_cnn.biases[1]
    x2 = e_cnn.act_fn_hid.(batchnorm(x1, e_cnn.bn_vars[1], e_cnn.bn_vars[2]))
    y = e_cnn.act_fn.(conv4(e_cnn.weights[2],x2;stride=1) .+ e_cnn.biases[2])
    
    return y
    
end


mutable struct EncoderCNNLarge
   
    weights
    biases
    bn_vars
    act_fn
    act_fn_hid
    
end

function initEncoderCNNLarge(input_dim, hidden_dim, num_objects, act_fn, act_fn_hid)
    
    weights = Any[param(3,3,3,hidden_dim),
                  param(3,3,hidden_dim,hidden_dim), 
                  param(3,3,hidden_dim,hidden_dim),
                  param(3,3,hidden_dim, num_objects)]
    
    biases = Any[param0(1,1,hidden_dim, 1), 
                 param0(1,1,hidden_dim, 1),
                 param0(1,1,hidden_dim, 1),
                 param0(1,1,num_objects,1)]
    
    bn_vars = Any[bnmoments(), atype(bnparams(hidden_dim)),
                  bnmoments(), atype(bnparams(hidden_dim)),
                  bnmoments(), atype(bnparams(hidden_dim))]
    
    return EncoderCNNLarge(weights, biases, bn_vars, act_fn, act_fn_hid)
    
end

function (e_cnn::EncoderCNNLarge)(x)
    
    #w(w_x,w_y,in_ch,out_ch)
    x1 = conv4(e_cnn.weights[1],x; stride = 1, padding=1) .+ e_cnn.biases[1]    
    x2 = e_cnn.act_fn_hid.(batchnorm(x1, e_cnn.bn_vars[1], e_cnn.bn_vars[2]))
    
    x3 = conv4(e_cnn.weights[2],x2; stride = 1, padding=1) .+ e_cnn.biases[2]    
    x4 = e_cnn.act_fn_hid.(batchnorm(x3, e_cnn.bn_vars[3], e_cnn.bn_vars[4]))
    
    x5 = conv4(e_cnn.weights[3],x4; stride = 1, padding=1) .+ e_cnn.biases[3]
    x6 = e_cnn.act_fn_hid.(batchnorm(x5, e_cnn.bn_vars[5], e_cnn.bn_vars[6]))
    
    y = e_cnn.act_fn.(conv4(e_cnn.weights[4],x6;stride=1, padding=1) .+ e_cnn.biases[4])
    
    return y
    
end
struct EncoderCNNSmall
   
    weights
    bn_vars
    act_fn
    act_fn_hid
    
end

function initEncoderCNNSmall(input_dim, hidden_dim, num_objects, act_fn, act_fn_hid)
    
    weights = Any[param(10,10,3,hidden_dim), param(1,1,hidden_dim,num_objects)]
    bn_vars = Any[bnmoments(), atype(bnparams(hidden_dim))]
    
    return EncoderCNNSmall(weights, bn_vars, act_fn, act_fn_hid)
    
end

function (e_cnn::EncoderCNNSmall)(x)
    
    #w(w_x,w_y,in_ch,out_ch)
    x1 = conv4(e_cnn.weights[1],x; stride = 10)
    x2 = e_cnn.act_fn_hid.(batchnorm(x1, e_cnn.bn_vars[1], e_cnn.bn_vars[2]))
    y = e_cnn.act_fn.(conv4(e_cnn.weights[2],x2;stride=1))
    
    return y
    
end
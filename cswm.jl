using Knet
include("utils.jl")
include("layernorm.jl")
include("encodercnn.jl")
include("encodermlp.jl")
include("gnn.jl")

function energy(transition_model, state, action, next_state, trans)
    
    norm = 0.5 / (sigma^2)
    
    if trans
        
        diff = state - next_state
        
    else
        
        pred_trans = transition_model(state,action)
        diff = state + pred_trans - next_state
        
    end
    
    #Diff has shape => (2,5,100)
    #Result has shape => (100)
    
    a  = norm*sum((diff.^2), dims=1)[1,:,:]
    res = mean(a,dims=1)[1,:]    
    return res
    
    
end

mutable struct ContrastiveSWM
    
    obj_extractor::EncoderCNNSmall
    obj_encoder::EncoderMLP
    gnn::TransitionGNN
    sigma
    hinge
    
end

function initContrastiveSWMSmall(input_ch, hidden_dim, num_objects, embedding_dim, action_dim, sigma, hinge )

    obj_extractor = initEncoderCNNSmall(input_ch, hidden_dim ÷ 16, num_objects, sigm, relu)
    obj_encoder = initEncoderMLP(25, hidden_dim, embedding_dim, num_objects, relu)
    gnn = initTransitionGNN(embedding_dim, hidden_dim, action_dim, num_objects, false, false, relu)
    
    return ContrastiveSWM(obj_extractor, obj_encoder, gnn, sigma, hinge)
    
end

# Forward propagation with transition
function (m::ContrastiveSWM)(obs,action)
    
    # Extract objects
    objs = m.obj_extractor(obs)
    
    # Obtain embeddings
    state = m.obj_encoder(objs)
    
    #Transition    
    out = m.gnn(state,action)
    
    return out
    
end

# Forward propagation with transition
function (m::ContrastiveSWM)(obs,action)
    
    # Extract objects
    objs = m.obj_extractor(obs)
    
    # Obtain embeddings
    state = m.obj_encoder(objs)
    
    #Transition    
    out = m.gnn(state,action)
    
    return out
    
end

# Forward propagation without transition
function (m::ContrastiveSWM)(obs)
    
    # Extract objects
    objs = m.obj_extractor(obs)
    
    # Obtain embeddings
    state = m.obj_encoder(objs)
    
    return state
    
end

# Contrastive loss part
function (m::ContrastiveSWM)(obs,action,next_obs)
    
    # Extract object embeddings
    state = m(obs)
    next_state = m(next_obs)    
    
    # Sample negative state across episodes at random
    batch_size = size(obs,4)
    perm = rand(1:batch_size)
    neg_state = state[:,:,perm]
    
    # Pos loss
    pos_loss = energy(m.gnn, state, action, next_state,true)
    zero_mat = atype(zeros(size(pos_loss)))
    pos_loss = mean(pos_loss)
    
    # Neg loss
    neg_loss = max.(zero_mat, hinge .- energy(m.gnn, state, action,next_state,false))
    neg_loss = mean(neg_loss)

    loss = pos_loss + neg_loss
    
    return loss
    
end
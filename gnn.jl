function unsorted_segment_sum(tensor, segment_ids, num_segments)
    
    """
    Replication of Tensorflow's unsorted_segment_sum in Julia
    Needs avoidance of for loops for the future work
    Computes the sum along segments of a tensor 
    such that output[i] = j...data[j..] where the sum is over 
    tuples j... such that segment_ids[j..]  
    """
    
    #In 2D Shapes, we expect:
    # tensor =>  (512, 2000)
    # segment_ids => (2000)
    # num_segments = 500
    # result => (512,500)
    
    #Allocate the result container
    result_shape = (size(tensor,1), num_segments)
    results = atype(zeros(result_shape))
    
    for i=1:size(segment_ids,1)
        
        id = segment_ids[i]
        results[:,id] += tensor[:,i]
        
    end    
    
    return results 
    
end

function get_edge_list_fully_connected(batch_size, num_objects)
    
    #Create fully-connected adjacency matrix for single sample.
    #adj_full has shape 5x5 for 2D Shapes Dataset
    adj_full = ones(num_objects, num_objects)
    
    #Remove diagonal
    adj_full = adj_full - diagm(ones(num_objects)) 
    
    #Store nonzero indices in a list
    #Edge_list has shape (20,), consisting of CartesianIndexes
    edge_list = findall(x -> x != 0, adj_full)
    num_entries = size(edge_list,1)

    #Convert it to an array for further processing
    #Now we have edge_list which has shape (20,2)
    edge_list = hcat(getindex.(edge_list, 2), getindex.(edge_list,1))

    # Copy edge list with `batch_size` times 
    edge_list = repeat(edge_list, batch_size)
    
    #Prepare offset.
    offset = [0:num_objects:batch_size*num_objects - 1;]
    offset = repeat(offset,num_entries)
    offset = reshape(offset, batch_size,:)
    offset =  vcat(offset'...)
    
    #Add the offset to the edge list
    edge_list = edge_list .+ offset
    
    return edge_list'
    

end


struct EdgeMLP
    
    weights
    biases
    layer_norm
    act_fn
  
end


function initEdgeMLP(input_dim, hidden_dim, act_fn)
    
    weights = [param(hidden_dim,input_dim*2),param(hidden_dim, hidden_dim),param(hidden_dim,hidden_dim)]
    biases =  [param0(hidden_dim), param0(hidden_dim), param0(hidden_dim)]    
    layer_norm = LayerNorm(hidden_dim)
    
    return EdgeMLP(weights, biases, layer_norm, act_fn)
    
end


function (e_mlp::EdgeMLP)(x)
    
    x1 = e_mlp.act_fn.(e_mlp.weights[1] * x .+ e_mlp.biases[1])
    x2 = e_mlp.weights[2]* x1 .+ e_mlp.biases[2]
    x3 = e_mlp.act_fn.(e_mlp.layer_norm(x2))
    x4 = e_mlp.act_fn.(e_mlp.weights[3]*x3 .+ e_mlp.biases[3])
    
    return x4
    
end

struct NodeMLP
    
    weights
    biases
    layer_norm
    act_fn
    
end

function initNodeMLP(node_input_dim, hidden_dim, act_fn, input_dim)
    
    weights = [param(hidden_dim,node_input_dim),param(hidden_dim, hidden_dim),param(input_dim,hidden_dim )]
    biases =  [param0(hidden_dim), param0(hidden_dim), param0(input_dim)]    
    layer_norm = LayerNorm(hidden_dim)
    
    return NodeMLP(weights, biases, layer_norm, act_fn)
    
end


function (node_mlp::NodeMLP)(x)
    
    x1 = node_mlp.act_fn.(node_mlp.weights[1] * x .+ node_mlp.biases[1])
    x2 = node_mlp.weights[2]* x1 .+ node_mlp.biases[2]
    x3 = node_mlp.act_fn.(node_mlp.layer_norm(x2))
    x4 = node_mlp.act_fn.(node_mlp.weights[3]*x3 .+ node_mlp.biases[3])
    
    return x4
end

function edge_model(edge_mlp, source, target)
    
    #Edge model
    #Out: [2000,4] in 2D Shapes
    out = vcat(source,target)
    return edge_mlp(out)
    
end

function node_model(node_mlp, node_attr , edge_index, edge_attr)
    
    if edge_attr == nothing
        
        out = node_attr
        
    else
        
        row = edge_index[1,:]
        col = edge_index[2,:]
        num_segments = size(node_attr,2)        
        agg = unsorted_segment_sum(edge_attr, row, num_segments)
        out = vcat(node_attr,agg)
        
    end
    
    return node_mlp(out)
    
end

mutable struct TransitionGNN
    """GNN-based transition model"""
    
    edge_mlp
    node_mlp
    
    ignore_action
    copy_action
    action_dim
    embedding_dim
    
    edge_list
    batch_size
    
end

function initTransitionGNN(input_dim, hidden_dim, action_dim, num_objects, ignore_action, copy_action, act_fn)
    
    if ignore_action
        
        action_dim = 0
        
    end
    
    #Edge MLP
    edge_mlp = initEdgeMLP(input_dim, hidden_dim, act_fn)
    node_input_dim = hidden_dim + input_dim + action_dim
    
    #Node MLP
    node_mlp = initNodeMLP(node_input_dim, hidden_dim, act_fn, input_dim)
    
    edge_list = nothing
    batch_size = 0
    
    return TransitionGNN(edge_mlp, node_mlp, ignore_action, copy_action, action_dim, input_dim,edge_list, batch_size)
    
end


#Forward prop
function (t_gnn::TransitionGNN)(states, action)
    
    dimensions = size(states)
    embedding_dim = dimensions[1]
    num_nodes = dimensions[2]
    batch_size = dimensions[3]
    
    # states: [batch_size (B), num_objects, embedding_dim]
    # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
    node_attr = reshape(states, t_gnn.embedding_dim,batch_size*num_nodes)
    
    edge_attr  = nothing
    edge_index = nothing
    
    if num_nodes > 1
        
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if t_gnn.edge_list == nothing || t_gnn.batch_size != batch_size
            
            
            edge_index = get_edge_list_fully_connected(batch_size, num_nodes)
            t_gnn.edge_list = edge_index
            
        end

        row = edge_index[1,:]
        col = edge_index[2,:]

        edge_attr = edge_model(t_gnn.edge_mlp,node_attr[:,row], node_attr[:,col])
        
    end
    
    #If action is included, concat actions to node_attr
    
    if !t_gnn.ignore_action
        
        if t_gnn.copy_action
            
            action_vec = toOneHot(action,t_gnn.action_dim)
            
        else
            
            action_vec = toOneHot(action, t_gnn.action_dim * num_nodes)
              
        end
        
        #??????
        action_vec = reshape(action_vec, 4,500)
        
        #node_attr => ([500, 2])
        #action_vec.shape => (500,4)
        node_attr = vcat(node_attr, action_vec)

    end 
    
    node_attr = node_model(t_gnn.node_mlp, node_attr , edge_index, edge_attr)
    
    # [batch_size, num_nodes, hidden_dim]
    #2,500
    return reshape(node_attr, t_gnn.embedding_dim ,num_nodes, batch_size ) # (2,5,100)
    
end










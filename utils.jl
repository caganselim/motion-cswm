using HDF5
using Random

function loadh5file(DATASET_PATH)
    f_e = h5open(DATASET_PATH,"r")
    dict = read(f_e)
    close(f_e)
    return dict
end

struct StateTransitionDataset
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""
   
    experience_buffer;
    # Build table for conversion between linear idx -> episode/step idx
    idx2episode;
    
    #Container to hold total number of steps
    num_steps;
    
    #Read array
    batch_idxs;

    #Batch size
    batch_size;
    
end

function buildDataset(DATASET_PATH, d_shuffle, batch_size)
    
    experience_buffer = loadh5file(DATASET_PATH)   
    step = 0
    
    println("Dataset loaded. Building dataset indexing...")
    
    idx2episode = []
    
    
    for ep in 1:length(experience_buffer)
        
        ep_key = string(ep-1)
        num_steps = length(experience_buffer[ep_key]["action"])
        
        for i in 1:num_steps
           
            push!(idx2episode,(ep_key,i))
            
        end 
        
        step += num_steps
        
    end
         
    batch_idxs = collect(1:step)
    
    if d_shuffle
        batch_idxs = shuffle(batch_idxs)
    end
    
    println("Done.")
        
    
    return  StateTransitionDataset(experience_buffer,idx2episode,step, batch_idxs, batch_size)
    
end

function toOneHot(idxs, dim)
    
    batch_size = length(idxs)
    hot_matrix = zeros(dim,batch_size)
    
    for i in 1:batch_size
        
        hot_matrix[idxs[i],i] = 1.0
        
    end
    
    return atype(hot_matrix)
    
end

function getitem(s,idx)
    
    ep_key, step = s.idx2episode[idx]
    obs = s.experience_buffer[ep_key]["obs"][:,:,:,step]
    action = s.experience_buffer[ep_key]["action"][step]
    next_obs = s.experience_buffer[ep_key]["next_obs"][:,:,:,step]
    
    return obs,action,next_obs
    
end

(s::StateTransitionDataset)(idx) = getitem(s,idx)

function prepareBatch(s,idx_1, idx_2)
    """Lazy loader to GPU."""
    
    minibatch = s.batch_idxs[idx_1:idx_2]
    minibatch_batch_size = size(minibatch,1)
    #print(minibatch_batch_size)
    
    #Read   
    b_obs = zeros(50,50,3,minibatch_batch_size)
    b_next_obs = zeros(50,50,3,minibatch_batch_size)
    b_action = zeros(minibatch_batch_size)
    
    for i in 1:minibatch_batch_size
        
        idx = minibatch[i]
        obs, action, next_obs = s(idx)
        
        #Insert obs
        b_obs[:,:,:,i] = obs
        
        #Assign action
        b_action[i] = action + 1
        
        #Insert next_obs
        b_next_obs[:,:,:,i] =  next_obs
        
    end
    
    return atype(b_obs), Integer.(b_action), atype(b_next_obs)
    
end

function iterate(s::StateTransitionDataset, state = 1)
    
    net_threshold = s.num_steps ÷ s.batch_size
        
    idx_1 = 1 + BATCH_SIZE*(state-1)
    idx_2 = BATCH_SIZE*state

    obs, action, next_obs = prepareBatch(s, idx_1,idx_2)
    
    if state == net_threshold
        
        state = 1
        
    end

    state += 1
    
    return ((obs, action, next_obs), state)
end
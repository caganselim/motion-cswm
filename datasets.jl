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

function buildStateTransitionDataset(dataset_path, d_shuffle, batch_size)
    
    experience_buffer = loadh5file(dataset_path)   
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


function getitem(s,idx)
    
    ep_key, step = s.idx2episode[idx]
    obs = s.experience_buffer[ep_key]["obs"][:,:,:,step]
    action = s.experience_buffer[ep_key]["action"][step]
    next_obs = s.experience_buffer[ep_key]["next_obs"][:,:,:,step]
    
    return obs,action,next_obs
    
end

(s::StateTransitionDataset)(idx) = getitem(s::StateTransitionDataset,idx)

function prepareBatch(s::StateTransitionDataset,idx_1, idx_2)
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
    remainder = s.num_steps % s.batch_size
    
    if state == -1
        return nothing
    end
    
    if state > net_threshold
        
        idx_1 = s.batch_size*(state-1) + 1
        idx_2 = s.num_steps
        #Set the state
        state = -1
        
    else
        
        idx_1 = 1 + s.batch_size*(state-1)
        idx_2 = s.batch_size*state
        
        #Increment the state
        
        if state == net_threshold
            
            state = -1
            
        else
            
            state += 1
            
        end
        
    end
    
    obs, action, next_obs = prepareBatch(s, idx_1,idx_2)
    
    return ((obs, action, next_obs), state)
end

##################Path Dataset##########################

struct PathDataset
    
    """
    Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """
    experience_buffer
    dataset_size
    batch_size
    
end

function buildPathDataset(dataset_path, batch_size)
    
    
    experience_buffer = loadh5file(dataset_path)
    dataset_size = length(experience_buffer)
    
    return PathDataset(experience_buffer, dataset_size, batch_size)
    
end

function getitem(p::PathDataset,idx)
    
    obs = p.experience_buffer[string(idx)]["obs"][:,:,:,1]
    action = p.experience_buffer[string(idx)]["action"][1]
    next_obs = p.experience_buffer[string(idx)]["next_obs"][:,:,:,1]

    return obs, action, next_obs
    
end

(p::PathDataset)(idx) = getitem(p::PathDataset,idx)

function prepareBatch(p::PathDataset, state)
    """Lazy loader to GPU."""
    
    #Convert state to index
    offset = (state - 1)*p.batch_size
    
    #Read 
    b_obs = zeros(50,50,3,p.batch_size)
    b_next_obs = zeros(50,50,3,p.batch_size)
    b_action = zeros(p.batch_size)
    
    for i in 1:p.batch_size
        
        idx = i + offset
        obs, action, next_obs = p(idx - 1)
        
        #Insert obs
        b_obs[:,:,:,i] = obs
        
        #Assign action
        b_action[i] = action + 1
        
        #Insert next_obs
        b_next_obs[:,:,:,i] =  next_obs
        
    end
    
    return atype(b_obs), Integer.(b_action), atype(b_next_obs)
    
end


function iterate(p::PathDataset, state = 1)
    
    net_threshold = p.dataset_size ÷ p.batch_size

    if state > net_threshold
        
        return nothing
        
    end  
        
    #Get evaluation batch
    obs, action, next_obs = prepareBatch(p, state)
    
    state += 1
    
    
    return ((obs, action, next_obs), state)
    
end

##################Multi Step Path Dataset##########################

struct MultiStepPathDataset
    
    """
    Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """
    experience_buffer
    dataset_size
    batch_size
    step_size
    
end

function buildMultiStepPathDataset(dataset_path, batch_size, step_size)
    
    
    experience_buffer = loadh5file(dataset_path)
    dataset_size = length(experience_buffer)
    
    return MultiStepPathDataset(experience_buffer, dataset_size, batch_size, step_size)
    
end

function getitem(p::MultiStepPathDataset,idx)
    
    obs = p.experience_buffer[string(idx)]["obs"][:,:,:,1]
    action = p.experience_buffer[string(idx)]["action"][1:p.step_size]
    next_obs = p.experience_buffer[string(idx)]["next_obs"][:,:,:,p.step_size]

    return obs, action, next_obs
    
end

(p::MultiStepPathDataset)(idx) = getitem(p::MultiStepPathDataset,idx)

function prepareBatch(p::MultiStepPathDataset, state)
    """Lazy loader to GPU."""
    
    #Convert state to index
    offset = (state - 1)*p.batch_size
    
    #Read 
    b_obs = zeros(50,50,3,p.batch_size)
    b_next_obs = zeros(50,50,3,p.batch_size)
    b_action = zeros(p.batch_size, p.step_size)
    
    for i in 1:p.batch_size
        
        idx = i + offset
        obs, action, next_obs = p(idx - 1)
        
        #Insert obs
        b_obs[:,:,:,i] = obs
        
        #Assign action
        b_action[i,:] = action .+ 1
        
        #Insert next_obs
        b_next_obs[:,:,:,i] =  next_obs
        
    end
    
    return atype(b_obs), Integer.(b_action), atype(b_next_obs)
    
end


function iterate(p::MultiStepPathDataset, state = 1)
    
    net_threshold = p.dataset_size ÷ p.batch_size

    if state > net_threshold
        
        return nothing
        
    end  
        
    #Get evaluation batch
    obs, action, next_obs = prepareBatch(p, state)
    
    state += 1
    
    
    return ((obs, action, next_obs), state)
    
end
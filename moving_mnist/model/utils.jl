using HDF5
using Random

function toOneHot(idxs, dim)
    
    batch_size = length(idxs)
    hot_matrix = zeros(dim,batch_size)
    
    for i in 1:batch_size
        
        hot_matrix[idxs[i],i] = 1.0
        
    end
    
    return atype(hot_matrix)
    
end
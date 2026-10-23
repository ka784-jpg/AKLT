# ==============================================================================
# 1. WINDOW CHECKER (Renamed from record_windowed to match your helper section)
# ==============================================================================
"""
Returns a function that records only when:
1. The step is within [ti, tf]
2. The step matches the interval starting from ti
"""
function window_checker(ti, tf, interval)
    return function(ctx)
        # Only record at the end of a complete step
        if !ctx.is_step_boundary
            return false
        end

        step = ctx.step_idx
        
        # Check range
        if step < ti || step > tf
            return false
        end

        # Check interval alignment
        return (step - ti) % interval == 0
    end
end

# ==============================================================================
# 2. OBSERVABLES
# ==============================================================================
# A new observable that wraps multiple StringOrder measurements
struct SpatiallyAveragedStringOrder <: AbstractObservable
    sub_observables::Vector{StringOrder}
end

# Constructor: Automatically generates the list of shifted pairs
function SpatiallyAveragedStringOrder(L::Int, num_shifts::Int, order::Int)
    obs_list = StringOrder[]
    half_L = L ÷ 2
    
    for k in 0:(num_shifts - 1)
        # Shift i and j by k to the right
        # mod1 ensures proper wrapping (1..L) for Periodic Boundary Conditions
        i = mod1(1 + k, L)
        j = mod1(half_L + 1 + k, L)
        
        # FIX: "order" passed as a keyword argument
        push!(obs_list, StringOrder(i, j; order=order))
    end
    return SpatiallyAveragedStringOrder(obs_list)
end

# The Functor: Calculates the mean of all sub-observables
function (avg_obs::SpatiallyAveragedStringOrder)(state)
    total_val = 0.0
    for obs in avg_obs.sub_observables
        total_val += obs(state) 
    end
    return total_val / length(avg_obs.sub_observables)
end

# ==============================================================================
# 3. HELPER: MEASUREMENT WORKER
# ==============================================================================
function record_step!(idx, t, state, L, self_av_space_it, SO_NN_mat, SO_NNN_mat, S_mat, bond_mat)
    # A. Measure Entropy (Scalar)
    S_val = EntanglementEntropy(cut=L÷2, renyi_index=1, base=2)(state)
    
    # Store [Time, Value]
    S_mat[idx, 1] = t
    S_mat[idx, 2] = S_val
    
    # B. Measure Bond Dimension
    # Store [Time, Value]
    bond_mat[idx, 1] = t
    bond_mat[idx, 2] = BondDimension()(state)
    
    # C. Measure Spatial Shifts
    # Set the first column to Time
    SO_NN_mat[idx, 1]  = t
    SO_NNN_mat[idx, 1] = t
    
    half_L = L ÷ 2
    for k in 0:(self_av_space_it - 1)
        # Shift indices
        i = mod1(1 + k, L)
        j = mod1(half_L + 1 + k, L)
        
        # Column Index: 1 is Time, so data starts at k + 2
        col_idx = k + 2
        
        # Measure and store directly
        SO_NN_mat[idx, col_idx]  = StringOrder(i, j; order=1)(state)
        SO_NNN_mat[idx, col_idx] = StringOrder(i, j; order=2)(state)
    end
end

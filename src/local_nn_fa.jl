mutable struct LocalNNFunctionApproximator{NT<:NNTree, V<:AbstractVector{Float64}} <: LocalFunctionApproximator
    nntree::NT
    nnpoints::AbstractVector{V}
    nnvalues::Vector{Float64}
    knnK::Union{Int64,Void}
    rnnR::Union{Float64,Void}
end
# Two default constructors for k and r
LocalNNFunctionApproximator(nntree::NT, nnpts::AbstractVector{V}, knnK::Int64) where {NT <: NNTree,V<:AbstractVector{Float64}} =
    LocalNNFunctionApproximator(nntree, nnpts, zeros(length(nntree.indices)), knnK, nothing)
LocalNNFunctionApproximator(nntree::NT, nnpts::AbstractVector{V}, rnnR::Float64) where {NT <: NNTree,V<:AbstractVector{Float64}} =
    LocalNNFunctionApproximator(nntree, nnpts, zeros(length(nntree.indices)), nothing, rnnR)

################ INTERFACE FUNCTIONS ################
function n_interpolants(nnfa::LocalNNFunctionApproximator)
    return length(nnfa.nntree.indices)
end

function get_all_interpolating_points(nnfa::LocalNNFunctionApproximator)
  return nnfa.nnpoints
end

function get_all_interpolating_values(nnfa::LocalNNFunctionApproximator)
  return nnfa.nnvalues
end

function get_interpolating_nbrs_idxs_wts(nnfa::LocalNNFunctionApproximator, v::AbstractVector{Float64})

    @assert (nnfa.knnK != nothing || nnfa.rnnR != nothing)

    if nnfa.knnK != nothing
        # Do k-NN lookup to get data and distances
        idxs, dists = knn(nnfa.nntree, v, nnfa.knnK)
    else
        # Do inrange lookup to get data
        # Then use metric to get dists between query and each nearest neighbor
        idxs = inrange(nnfa.nntree, v, nnfa.rnnR)
        dists = zeros(length(idxs))
        for (i,idx) in enumerate(idxs)
            dists[i] = Distances.evaluate(nnfa.nntree.metric, v, nnfa.nnpoints[idx])
        end
    end

    probs = zeros(length(dists))

    # If exactly one point, set that probability of that idx to 1
    # and that of all others to 0.
    if minimum(dists) < eps(Float64)
        probs[indmin(dists)] = 1.0
    else
        for (i,d) in enumerate(dists)
            probs[i] = 1.0/d
        end
        probs /= sum(probs)
    end

    return (idxs, probs)
end


function evaluate(nnfa::LocalNNFunctionApproximator, v::AbstractVector{Float64})

    idxs, wts = get_interpolating_nbrs_idxs_wts(nnfa, v)

    # Do a weighted average of values
    value::Float64 = 0.0
    wtsum::Float64 = 0.0
    for (i,idx) in enumerate(idxs)
        value += wts[i]*nnfa.nnvalues[idx]
        wtsum += wts[i]
    end
    value /= wtsum

    return value
end

function batch_update(nnfa::LocalNNFunctionApproximator, nnvalues::AbstractVector{Float64})
  nnfa.nnvalues = deepcopy(nnvalues)
end

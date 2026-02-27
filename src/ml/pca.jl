function prepareDataPCA(data::InteractionData)
   # intialise arrays
    protein = getProteinData(data, 1)
    dimensions = getSampleMatrix(protein)
    dimensions = size(dimensions,1) * size(dimensions,2) * size(dimensions,3)
    nproteins = length(getIDs(data[1]["controls"]))

    values = zeros(Union{Float64, Missing}, dimensions * 2, nproteins)
    values .= missing

    # iterate over data and populate arrays
    for i in 1:nproteins
        protein = getProteinData(data, i)
        x = getSampleMatrix(protein)
        x_flatten::Vector{Union{Float64, Missing}} = [x[j] for j ∈ eachindex(x)]

        y = getControlMatrix(protein)
        y_flatten::Vector{Union{Float64, Missing}} = [y[j] for j ∈ eachindex(y)]
        append!(x_flatten,y_flatten)
        values[:, i] = x_flatten
    end

    sample_type = fill("sample", dimensions)
    append!(sample_type, fill("control", dimensions))

    # remove dummy data columns, i.e. columns with only Missing
    θ = sum(ismissing.(values), dims = 2)[:,1]
    positions_all_missing = findall(θ .== nproteins)
    deleteat!(sample_type, positions_all_missing)
    values = values[setdiff(1:2*dimensions, positions_all_missing),:]

     # remove missing values
     values[ismissing.(values)] .= 0.0
     values_sparse = zeros(size(values)...); values_sparse .= values
    return sample_type, values_sparse#SparseArrays.sparse(values_sparse)
end


# define new method for fit 
function fit(
    ::Type{PPCA}, data::InteractionData;
    method::Symbol = :bayes, maxoutdim::Int = 2, mean = nothing, tol = 1e-06, maxiter = 1000
)
    # construct data matrix
    pcalabels, pcadata = prepareDataPCA(data)

    # fit pca
    return (
        pca_result = fit(
            PPCA, pcadata; 
            method = method, maxoutdim = maxoutdim, 
            mean = mean, tol = tol, maxiter = maxiter), 
        pca_labels = pcalabels
        )
end

function fit(
    ::Type{PCA}, data::InteractionData;
    method::Symbol = :auto, maxoutdim::Int = 2, pratio = 0.99,  mean = nothing
)
    #check arguments
    in(method,[:auto, :cov, :svd]) || @error "method must be :auto, :cov or :svd"
    maxoutdim > 0 || @error "maxoutdim must be > 0"

    # construct data matrix
    pcalabels, pcadata = prepareDataPCA(data)

    # fit pca
    return (
        pca_result = fit(
            PCA, pcadata; 
            method = method, maxoutdim = maxoutdim, 
            mean = mean, pratio = pratio), 
        pca_labels = pcalabels
        )
end



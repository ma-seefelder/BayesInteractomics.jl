module BayesInteractomicsEmbeddingsExt

using BayesInteractomics
using UMAP
using Clustering

import Random
import Distances
import LinearAlgebra

# Re-export of core symbols the included files need to refer to.
import BayesInteractomics: fit_sample_umap, fit_protein_umap, fit_condition_clustering, fit_sample_tsne

include("sample_umap.jl")
include("protein_umap.jl")
include("hclust_helper.jl")

# TSne is a declared trigger of this extension (see Project.toml [extensions]); it is therefore
# guaranteed present whenever this module loads. The guard below is a defensive no-op-if-absent.
@static if Base.find_package("TSne") !== nothing
    using TSne
    include("sample_tsne.jl")
end

end # module BayesInteractomicsEmbeddingsExt

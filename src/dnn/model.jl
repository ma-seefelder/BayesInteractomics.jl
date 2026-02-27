
# This block runs only when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg; Pkg.activate("./src/dnn/");
    import Flux
    using CUDA
    CUDA.allowscalar(false)
end

# ------------------------------------------------------------------------------ #
# Model definition
# ------------------------------------------------------------------------------ #

"""
    getDNNModel(
        n_layers::Int = 4,
        n_neurons::Vector{Int} = [1024, 512, 256, 128], # Input to first layer, then output of intermediate layers
        activation::Vector{Union{Any,Function}} = Any[Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
        dropout_rate::Float64 = 0.1
    )

Constructs a dense neural network model using Flux.jl.

The network consists of `n_layers` blocks, each comprising a `Dense` layer,
a `BatchNorm` layer, an activation function, and a `Dropout` layer.
The architecture is defined by the `n_neurons` vector, which dictates the
input and output dimensions of the `Dense` layers, and the `activation` vector
for their respective activation functions. The final `Dense` layer in the
constructed chain always outputs a single neuron.

# Arguments
- `n_layers::Int = 4`: The number of `Dense` layer blocks in the network. This must match the length of the initial `n_neurons` vector and the `activation` vector.
- `n_neurons::Vector{Int} = [1024, 512, 256, 128]`: A vector of integers defining neuron counts.
    - `n_neurons[1]`: Input dimension to the first layer.
    - For `1 <= i < n_layers`, `n_neurons[i+1]` is the output dimension of `Dense` layer `i` and input dimension to `Dense` layer `i+1`.
    - `n_neurons[n_layers]` is the input dimension to the `n_layers`-th `Dense` layer.
    The `n_layers`-th `Dense` layer will output 1 neuron.
- `activation::Vector{Union{Any,Function}} = Any[Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid]`: A vector of activation functions. `activation[i]` is applied after the `i`-th `BatchNorm` layer.
- `dropout_rate::Float64 = 0.1`: The dropout rate applied after each activation function. If `activation[1]` is `Flux.selu`, `Flux.AlphaDropout` is used; otherwise, `Flux.Dropout` is used.

# Returns
- `Flux.Chain`: A Flux model (`Chain`) representing the constructed deep neural network. The chain will consist of `n_layers` blocks, each containing `Dense -> BatchNorm -> activation_function -> Dropout`.

# Details
The function validates that `n_layers` matches the lengths of the `n_neurons`
and `activation` vectors. It then appends `1` to the `n_neurons` vector
(mutating it if it was passed as an argument from outside the function's defaults)
to define the output size of the final `Dense` layer. The model is constructed as a
`Flux.Chain`. Each of the `n_layers` blocks consists of a `Flux.Dense` layer,
followed by `Flux.BatchNorm`, the specified activation function, and a dropout layer
(`Flux.AlphaDropout` if the first activation is `Flux.selu`, `Flux.Dropout` otherwise).

# Examples
```julia
# Default 4-layer model (1024 -> 512 -> 256 -> 128 -> 1)
model1 = getDNNModel()

# Custom 2-layer model (784 -> 64 -> 1)
model2 = getDNNModel(2, [784, 64], [Flux.tanh, Flux.sigmoid])
```

# Throws
- `ArgumentError`: If `n_layers` does not match the length of `n_neurons` or `activation`.
"""
function getDNNModel(
    n_layers::Int = 4, 
    n_neurons::Vector{Int} = [1024, 512, 256, 128],
    activation::Vector{Union{Any,Function}} = Any[Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    dropout_rate::Float64 = 0.1
    )
    # check arguments
    n_layers == length(n_neurons) || throw(ArgumentError("n_layers and n_neurons must have the same length"))
    n_layers == length(activation) || throw(ArgumentError("n_layers and activation must have the same length"))

    push!(n_neurons, 1)

    layers = []
    for i in 1:n_layers
        push!(layers, Flux.Dense(n_neurons[i], n_neurons[i+1]))
        i >= 2 && i < n_layers && push!(layers, Flux.BatchNorm(n_neurons[i+1]))
        push!(layers, activation[i])
        dropout_rate > 0.0 && i < n_layers ? push!(layers, Flux.Dropout(dropout_rate)) : nothing
    end
    return Flux.Chain(layers...)
end


"""
    activation_to_string(act_func)
    Converts a Flux activation function name to a string for logging.
"""
function activation_to_string(act_func)
    if act_func == Flux.relu
        return "relu"
    elseif act_func == Flux.sigmoid
        return "sigmoid"
    elseif act_func == Flux.tanh
        return "tanh"
    elseif act_func == Flux.elu
        return "elu"
    elseif act_func == Flux.leakyrelu
        return "leakyrelu"
    elseif act_func == Flux.swish
        return "swish"
    elseif act_func == Flux.celu
        return "celu"
    elseif act_func == Flux.selu
        return "selu"
    elseif act_func == Flux.gelu
        return "gelu"
    else
        return string("")
    end
end


function _define_layers(n_neurons, n_layers; input_size = 3072) 
    neurons = [input_size]
    for _ in 1:n_layers-1
        push!(neurons, n_neurons)
    end
    return neurons
end

function _define_activations(activation, n_layers) 
    if activation == "relu"
        activation = Flux.relu
    elseif activation == "sigmoid"
        activation = Flux.sigmoid
    elseif activation == "tanh"
        activation = Flux.tanh
    elseif activation == "elu"
        activation = Flux.elu
    elseif activation == "leakyrelu"
        activation = Flux.leakyrelu
    elseif activation == "swish"
        activation = Flux.swish
    elseif activation == "celu"
        activation = Flux.celu
    elseif activation == "selu"
        activation = Flux.selu
    elseif activation == "gelu"
        activation = Flux.gelu
    else
        @error "Unsupported activation function: $activation"
    end

    result = []
    for _ in 1:n_layers-1
        push!(result, activation)
    end
    push!(result, Flux.sigmoid)
    return result
end

## --------------------------------------------------------------- #
### Multi-head attention model ------------------------------------ #
### --------------------------------------------------------------- #

# Transformer block module

"""
    TransformerBlock(mha, norm1, ffn, norm2)
    TransformerBlock{MHA_TYPE, NORM_TYPE1, FFN_TYPE, NORM_TYPE2}(mha, norm1, ffn, norm2)

A standard Transformer encoder block, consisting of:
1. A Multi-Head Self-Attention layer (`mha`).
2. A residual connection and Layer Normalization (`norm1`).
3. A position-wise Feed-Forward Network (`ffn`).
4. A second residual connection and Layer Normalization (`norm2`).

# Fields
- `mha`: Multi-head self-attention layer.
- `norm1`: Layer normalization applied after the first residual connection.
- `ffn`: Position-wise feed-forward network.
- `norm2`: Layer normalization applied after the second residual connection.

The type parameters `MHA_TYPE`, `NORM_TYPE1`, `FFN_TYPE`, `NORM_TYPE2` correspond to the types of these fields.
"""
struct TransformerBlock{MHA, N1, FFN, N2}
    mha::MHA
    norm1::N1
    ffn::FFN
    norm2::N2
end

Flux.@layer TransformerBlock

"""
    (block::TransformerBlock)(x::AbstractArray{T, 3}) where T<:Real

Applies the Transformer block to the input `x`.

The input `x` is processed through a multi-head self-attention mechanism,
followed by a residual connection and layer normalization. Then, it passes
through a position-wise feed-forward network, again followed by a residual
connection and layer normalization.

# Arguments
- `x::AbstractArray{<:Real, 3}`: Input tensor of shape `(feature_dim, sequence_len, batch_size)`.

# Returns
- `AbstractArray{<:Real, 3}`: Output tensor of the same shape as `x`.
"""
function (block::TransformerBlock)(x::AbstractArray{<:Real, 3})
    # First sublayer: Multi-Head Attention (Pre-LN)
    normed_x_for_mha = block.norm1(x)
    attn_output, _ = block.mha(normed_x_for_mha, normed_x_for_mha, normed_x_for_mha)
    # Residual connection for the first sublayer
    x_after_mha = x .+ attn_output

    # Second sublayer: Feed-Forward Network (Pre-LN)
    normed_x_for_ffn = block.norm2(x_after_mha) # Normalize the output of the first sublayer
    ffn_output = block.ffn(normed_x_for_ffn)
    # Residual connection for the second sublayer
    x_after_ffn = x_after_mha .+ ffn_output
    return x_after_ffn
end

# complete model
"""
    ProteinPairTransformerModel{E, DC}

A neural network model designed for protein-protein interaction prediction.
It processes a pair of protein encodings by concatenating them into a sequence,
passing this sequence through an encoder chain (typically Transformer blocks),
pooling the resulting sequence representation, and finally processing the
pooled features through a dense neural network.

The model expects input `x` to be a matrix where columns are samples and rows
are features. Each sample's features should be the concatenation of two protein
encodings, each of `protein_feature_dim` dimensions.

# Fields
- `protein_feature_dim::Int`: The dimensionality of a single protein's feature encoding. Used to split the input and defines the feature dimension for the encoder.
- `encoder_chain::E`: A `Flux.Chain` or similar structure containing the sequence processing layers (e.g., Transformer Blocks). It operates on the concatenated sequence of protein encodings.
- `dense_chain::DC`: A `Flux.Chain` of dense layers that processes the pooled output of the encoder chain.

The type parameters `E` and `DC` correspond to the types of the `encoder_chain` and `dense_chain` fields, respectively.
"""
struct ProteinPairTransformerModel{E, DC}
    protein_feature_dim::Int
    encoder_chain::E 
    dense_chain::DC
end

Flux.@layer ProteinPairTransformerModel

"""
    (m::ProteinPairTransformerModel)(x::AbstractMatrix)

Performs the forward pass for the `ProteinPairTransformerModel`.

   (m::ProteinPairTransformerModel)(x::AbstractMatrix{<:Real})

Performs the forward pass for the `ProteinPairTransformerModel`.

The input `x` is expected to be a matrix where each column is a sample,
and the rows represent concatenated feature encodings of two proteins.
Specifically, if `m.protein_feature_dim` is `D`, then `x` should have
`2*D` rows. The first `D` rows are for protein A, and the next `D` rows are for protein B.
are for protein B.

The function performs bidirectional cross-attention (A queries B, B queries A),
sums the attention outputs, flattens the result, and passes it through
the model's `dense_chain`.

# Arguments
- `x::AbstractMatrix`: Input matrix of size `(2 * m.protein_feature_dim, batch_size)`.

# Returns
- `AbstractMatrix`: Output predictions from the dense chain, typically of size `(1, batch_size)`.
"""
function (m::ProteinPairTransformerModel)(x::AbstractMatrix) # x is (2*protein_feature_dim, batch_size)
    batch_s = size(x, 2)

    idx_split = m.protein_feature_dim
    x_a = x[1:idx_split, :]
    x_b = x[(idx_split + 1):(2*idx_split), :]

    # Reshape inputs for the MultiHeadAttention layer.
    # MHA expects (feature_dim, sequence_length=1, batch_size)
    x_a_reshaped = reshape(x_a, (m.protein_feature_dim, 1, batch_s))
    x_b_reshaped = reshape(x_b, (m.protein_feature_dim, 1, batch_s))
    input_sequence = cat(x_a_reshaped, x_b_reshaped, dims = 2)
    
    # Perform attention mechanism
    encoded_sequence = m.encoder_chain(input_sequence)
    attn_flat = Flux.flatten(encoded_sequence) 
    output = m.dense_chain(attn_flat)
    return output
end

function _validate_mha_model_args(
    protein_feature_dim::Int,
    n_heads::Int,
    post_attention_n_layers::Int,
    post_attention_n_neurons::Vector{Int},
    post_attention_activation::Vector{Union{Any,Function}}
)
    if !(protein_feature_dim > 0)
        throw(ArgumentError("protein_feature_dim must be positive."))
    end
    if !(n_heads > 0)
        throw(ArgumentError("n_heads must be positive."))
    end
    if protein_feature_dim % n_heads != 0
        throw(ArgumentError("protein_feature_dim ($protein_feature_dim) must be divisible by n_heads ($n_heads)."))
    end
    if length(post_attention_n_neurons) != post_attention_n_layers
        throw(ArgumentError("Length of post_attention_n_neurons ($(length(post_attention_n_neurons))) must match post_attention_n_layers ($post_attention_n_layers)."))
    end
    if length(post_attention_activation) != post_attention_n_layers
        throw(ArgumentError("Length of post_attention_activation ($(length(post_attention_activation))) must match post_attention_n_layers ($post_attention_n_layers)."))
    end
    return true
end

"""
    getMultiHeadAttentionModel(
        protein_feature_dim::Int = 512,
        n_attention_layers::Int = 2,
        n_heads::Int = 8,
        attention_dropout_rate::Float64 = 0.1,
        post_attention_n_layers::Int = 4,
        post_attention_n_neurons::Vector{Int} = [512, 256, 128, 64],
        post_attention_activation::Vector{Union{Any,Function}} = [Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
        post_attention_dropout_rate::Float64 = 0.1
    )

Constructs a `ProteinPairTransformerModel` neural network for protein-protein
interaction prediction.

This model takes concatenated protein encodings as input, treats them as a
sequence of length 2, passes this sequence through a chain of Transformer
encoder blocks, applies mean pooling to the output sequence, and finally
processes the pooled features through a dense neural network to produce a
single output prediction.

# Arguments
- `protein_feature_dim::Int = 512`: The dimensionality of each individual protein encoding.
  This also defines the input and output dimension of the multi-head attention block.
- `n_attention_layers::Int = 2`: The number of Transformer encoder blocks in the model.
- `n_heads::Int = 8`: The number of attention heads in the `MultiHeadAttention` layer.
  `protein_feature_dim` must be divisible by `n_heads`.
- `attention_dropout_rate::Float64 = 0.1`: Dropout probability for the `MultiHeadAttention` layer.
- `post_attention_n_layers::Int = 4`: The number of dense layer blocks following the
  attention mechanism. This must match the length of `post_attention_n_neurons`
  and `post_attention_activation`.
- `post_attention_n_neurons::Vector{Int} = [512, 256, 128, 64]`: A vector defining neuron counts
  for the dense layers.
    - `post_attention_n_neurons[1]` must be equal to `protein_feature_dim`, as it's the
      input dimension from the attention block to the first dense layer.
    - For `1 <= i < post_attention_n_layers`, `post_attention_n_neurons[i+1]` is the output
      dimension of dense layer `i` and input to dense layer `i+1`.
    - `post_attention_n_neurons[post_attention_n_layers]` is the input dimension to the
      `post_attention_n_layers`-th dense layer. This layer will output 1 neuron.
- `post_attention_activation::Vector{Union{Any,Function}} = [...]`: A vector of activation
  functions for the dense layers. `post_attention_activation[i]` is applied after the
  `i`-th dense layer's (optional) `BatchNorm`.
- `post_attention_dropout_rate::Float64 = 0.1`: Dropout rate applied after activation in
  the dense layers (from the second dense block onwards, and not for the final output layer).

# Returns
`ProteinPairTransformerModel`: An instance of the model, ready for training. The model structure is:
  Input (Concatenated Pair) -> Reshape to Sequence -> Encoder Chain (Transformer Blocks) -> Mean Pooling -> Flatten -> Dense Chain -> Output.

# Details
The dense chain following the attention mechanism is constructed similarly to `getDNNModel`.
Each of the `post_attention_n_layers` blocks typically consists of a `Flux.Dense` layer,
followed by `Flux.BatchNorm` (from the second block onwards, if not the final output neuron),
the specified activation function, and `Flux.Dropout` (from the second block onwards,
if not the final output neuron and `post_attention_dropout_rate > 0`).

# Throws
- `ArgumentError`: If `protein_feature_dim` or `n_heads` are not positive, if `protein_feature_dim` is not divisible by `n_heads`, if `post_attention_n_neurons[1]` does not match `protein_feature_dim`, or if `post_attention_n_layers` does not match the length of `post_attention_n_neurons` or `post_attention_activation`.
"""
function getProteinTransformerModel(
    protein_feature_dim::Int = 512,
    n_attention_layers::Int = 2,
    n_heads::Int = 8,
    attention_dropout_rate::Float64 = 0.1,
    post_attention_n_layers::Int = 4, 
    post_attention_n_neurons::Vector{Int} = [512, 256, 128, 64], 
    post_attention_activation::Vector{Union{Any,Function}} = Any[Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    post_attention_dropout_rate::Float64 = 0.1
    )

    # --- Argument Validations ---
    _validate_mha_model_args(
        protein_feature_dim,
        n_heads,
        post_attention_n_layers,
        post_attention_n_neurons,
        post_attention_activation
    )

    # Build the Transformer Encoder chain
    layers = []
    for _ in 1:n_attention_layers
        mha = Flux.MultiHeadAttention(
            protein_feature_dim, nheads=n_heads, dropout_prob=attention_dropout_rate
            )
        # Feed-forward layer
        ffn_dim = protein_feature_dim * 4 # Common practice to have a wider intermediate layer
        ffn = Flux.Chain(
            Flux.Dense(protein_feature_dim => ffn_dim, Flux.relu),
            Flux.Dense(ffn_dim => protein_feature_dim)
        )

        block = TransformerBlock(
            mha,
            Flux.LayerNorm(protein_feature_dim),
            ffn,
            Flux.LayerNorm(protein_feature_dim)
        )
    
        # Create the TransformerBlock and add it to the chain
        push!(layers, block)
        end

    encoder_chain = Flux.Chain(layers...)
    post_attention_n_neurons[1] = 1024 
    # --- Dense Layers Chain ---
    # Use getDNNModel to construct the dense part of the network.
    final_dense_chain = getDNNModel(
        post_attention_n_layers,
        post_attention_n_neurons,
        post_attention_activation,
        post_attention_dropout_rate
    )

    # --- Construct and return the model ---
    return ProteinPairTransformerModel(protein_feature_dim, encoder_chain, final_dense_chain)
end

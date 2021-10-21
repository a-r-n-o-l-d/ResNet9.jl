module ResNet9

using Flux
using Flux: convfilter, Zeros
using Base.Iterators: takewhile

export resnet9

conv(chs) = Conv((3, 3), chs, identity, pad = (1, 1), bias = false)

function convblock(chs::Pair; pool = false)
    _, out = chs
    cv = conv(chs)
    bn = BatchNorm(out, relu)
    pool && return cv, bn, MaxPool((2, 2))
    cv, bn
end

convblock(ch::Int) = convblock(ch=>ch)

resblock(ch) = SkipConnection(Chain(convblock(ch)..., convblock(ch)...), +)

function classifier(ch, nc, drp)
    fl = [GlobalMaxPool(), Flux.flatten]
    fc = []
    if drp > 0
         push!(fc, Dropout(drp))
    end
    (nc > 2) ? push!(fc, Dense(ch, nc), softmax) : push!(fc, Dense(ch, 1, sigmoid))
    fl..., fc...
end

"""
    resnet9(;inchannels, nclasses, basewidth = 64, expansion = 2)
Build a ResNet9 network.
* inchannels: number of input channels.
* nclasses: number of classes, if `nclasses` = 2 the network ends with a fully
  connected layer with a sigmoid activation function, otherwise it ends with a 
  fully connected layer followed by a softmax function.
* basewidth: base number of channels.
* expansion: factor of channels expansion.
"""
function resnet9(;inchannels, nclasses, dropout = 0, basewidth = 64, expansion = 2) # groupnorm
    ch1 = basewidth
    ch2 = expansion * ch1
    ch3 = expansion * ch2
    ch4 = expansion * ch3
    Chain(convblock(inchannels=>ch1)...,                      # Input layer
          convblock(ch1=>ch2, pool = true)..., resblock(ch2), # Layer 1
          convblock(ch2=>ch3, pool = true)...,                # Layer 2
          convblock(ch3=>ch4, pool = true)..., resblock(ch4), # Layer 3
          classifier(ch4, nclasses, dropout)...)              # Classifier
end

"""
    resnet9(model::Chain; nclasses, dropout)
Build a ResNet9 model from an existing (pre-trained) model and change the first 
or last layers according to `inchannels`, `nclasses` and `dropout`.
"""
function resnet9(model::Chain;
                 inchannels = inchan(model), 
                 nclasses = nclass(model),
                 dropout = dpout(model))
    ich = inchan(model)
    if inchannels != ich
        l = model[1]
        iout = size(l.weight, ndims(l.weight))
        model = Chain(conv(inchannels=>iout), model[2:end]...)
    end
    if nclasses != nclass(model)
        fc = fclayer(model)
        fcin = size(fc.weight, 2)
        i = (findall(l -> isa(l, GlobalMaxPool), model) |> first) - 1 #takewhile(l -> !isa(l, GlobalMaxPool), model) |> collect
        back = model[1:i]
        front = classifier(fcin, nclasses, dropout)
        model = Chain(back..., front...)
    elseif dropout != dpout(model)
        i = (findall(l -> isa(l, GlobalMaxPool), model) |> first) + 1
        j = findall(l -> isa(l, Dense), model) |> first
        back = model[1:i]
        front = model[j:end]
        front = (dropout > 0) ? [Dropout(dropout), front...] : front
        model = Chain(back..., front...)
    end
    model
end

inchan(m) = size(m[1].weight, ndims(m[1].weight) - 1)

fclayer(m) = filter(l -> isa(l, Dense), m.layers) |> first

nclass(m) = fclayer(m) |> l -> size(l.weight, 1)

dpout(m) = m[isa.(m, Dropout)] |> c -> (isempty(c.layers)) ? 0 : c.layers[1].p

# """
#     resnet9(model::Chain; nclasses, dropout)
# Build a ResNet9 model from an existing (pre-trained) model and change the first
# layer according to `inchannels`.
# """
# function resnet9(model::Chain; inchannels)
#     ch = nchout(model[1:2])
#     Chain(convblock(inchannels=>ch)..., model[3:end]...)
# end

# function modelarch(model::Chain)
#     l = model[1]
#     ich = size(l.weight, ndims(l.weight) - 1)
#     fcin, fcout = 0, 0
#     for l ∈ model
#         if isa(l, Dense)
#             fcin = size(l.weight, 2)
#             fcout = size(l.weight, 1)
#         end
#     end
#     ich, fcin, fcout
# end

# function ncl(m::Chain)
#     # for l ∈ 
#     #     takewhile(l -> !isa(l, Dense), m)

#     # size(d.weight, 1)
# end

# nchin(l) = 0
# nchin(l::Conv) = size(l.weight, ndims(l.weight) - 1)
# nchin(l::BatchNorm) = l.chs
# nchin(l::SkipConnection) = nchin(l.layers)
# function nchin(c::Chain)
#     nch = 0
#     for l ∈ c
#         nch = nchin(l)
#         if nch > 0
#             break
#         end
#     end
#     if nch == 0
#         error("Failed to find number of channels.") |> throw
#     end
#     nch
# end

# nchout(l) = 0
# nchout(l::Conv) = size(l.weight, ndims(l.weight))
# nchout(l::BatchNorm) = l.chs
# nchout(l::SkipConnection) = nchout(l.layers)
# function nchout(c::Chain)
#     layers = [c...] |> reverse
#     nch = 0
#     for l ∈ layers
#         nch = nchout(l)
#         if nch > 0
#             break
#         end
#     end
#     if nch == 0
#         error("Failed to find number of channels.") |> throw
#     end
#     nch
# end

end
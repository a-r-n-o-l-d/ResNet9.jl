module ResNet9

using Flux
using Flux: convfilter, Zeros
using Base.Iterators: takewhile

export resnet9

function convblock(chs::Pair; pool = false)
    _, out = chs
    cv = Conv((3, 3), chs, identity, pad = (1, 1), bias = false)
    bn =  BatchNorm(out, relu)
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
    #fc = (nc > 2) ? [Dense(ch, nc), softmax] : [Dense(ch, 1, sigmoid)]
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
Build a ResNet9 model from an existing (pre-trained) model and change the last
layers according to `nclasses` and `dropout`.
"""
function resnet9(model::Chain; nclasses, dropout)
    back = takewhile(l -> !isa(l, GlobalMaxPool), model) |> collect
    ch = nchout(model)
    front = classifier(ch, nclasses, dropout)
    Chain(back..., front...)
end

"""
    resnet9(model::Chain; nclasses, dropout)
Build a ResNet9 model from an existing (pre-trained) model and change the first
layer according to `inchannels`.
"""
function resnet9(model::Chain; inchannels)
    ch = nchout(model[1:2])
    Chain(convblock(inchannels=>ch)..., model[3:end]...)
end

nchout(l) = 0
nchout(l::Conv) = size(l.weight, ndims(l.weight))
nchout(l::BatchNorm) = l.chs
nchout(l::SkipConnection) = nchout(l.layers)
function nchout(c::Chain)
    layers = [c...] |> reverse
    nch = 0
    for l ∈ layers
        nch = nchout(l)
        if nch > 0
            break
        end
    end
    if nch == 0
        error("Failed to find number of channels.") |> throw
    end
    nch
end

end
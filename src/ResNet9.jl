module ResNet9

using Flux
using Flux: convfilter, Zeros

export resnet9

function convblock(f::Pair; pool = false)
    _, out = f
    cv = Conv((3, 3), f, identity, pad = (1, 1), bias = false)
    bn =  BatchNorm(out, relu)
    pool && return cv, bn, MaxPool((2, 2))
    cv, bn
end

convblock(f::Int) = convblock(f=>f)

resblock(f) = SkipConnection(Chain(convblock(f)..., convblock(f)...), +)

function classifier(f, n)
    fl = (GlobalMaxPool(), Flux.flatten)
    n > 2 && return fl..., Dense(f, n), softmax
    fl..., Dense(f, 1, sigmoid)
end

function resnet9(;ichs, ncls, flts = 64) # dropout, groupnorm, bson file resnet9!
    f1 = flts
    f2 = 2 * f1
    f3 = 2 * f2
    f4 = 2 * f3
    Chain(convblock(ichs=>f1)...,                          # Input layer
          convblock(f1=>f2, pool = true)..., resblock(f2), # Layer 1
          convblock(f2=>f3, pool = true)...,               # Layer 2
          convblock(f3=>f4, pool = true)..., resblock(f4), # Layer 3
          classifier(f4, ncls)...)                         # Classifier
end

end
module ResNet9

using Flux
using Flux: convfilter, Zeros

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

function classifier(ch, nc)
    fl = (GlobalMaxPool(), Flux.flatten)
    #n > 1 && return fl..., Dense(f, n), softmax
    fc = (nc > 2) ? [Dense(ch, nc), softmax] : [Dense(ch, 1, sigmoid)]
    fl..., fc...
end

"""
    resnet9(;inchannels, nout, basewidth = 64)
"""
function resnet9(;inchannels, nclasses, basewidth = 64) # dropout, groupnorm, bson file resnet9! expansion
    ch1 = basewidth
    ch2 = 2 * ch1
    ch3 = 2 * ch2
    ch4 = 2 * ch3
    Chain(convblock(inchannels=>ch1)...,                      # Input layer
          convblock(ch1=>ch2, pool = true)..., resblock(ch2), # Layer 1
          convblock(ch2=>ch3, pool = true)...,                # Layer 2
          convblock(ch3=>ch4, pool = true)..., resblock(ch4), # Layer 3
          classifier(ch4, nclasses)...)                       # Classifier
end

end
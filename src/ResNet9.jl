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
    resnet9(;inchannels, nclasses, basewidth = 64, expansion = 2)
Build a ResNet9 network.
* inchannels: number of input channels.
* nclasses: number of classes, if `nclasses` = 2 the network ends with a fully
  connected layer with a sigmoid activation function, otherwise it ends with a 
  fully connected layer followed by a sftmax function.
* basewidth: base number of channels.
* expansion: factor of channels expansion.
"""
function resnet9(;inchannels, nclasses, basewidth = 64, expansion = 2) # dropout, groupnorm, bson file resnet9! expansion
    ch1 = basewidth
    ch2 = expansion * ch1
    ch3 = expansion * ch2
    ch4 = expansion * ch3
    Chain(convblock(inchannels=>ch1)...,                      # Input layer
          convblock(ch1=>ch2, pool = true)..., resblock(ch2), # Layer 1
          convblock(ch2=>ch3, pool = true)...,                # Layer 2
          convblock(ch3=>ch4, pool = true)..., resblock(ch4), # Layer 3
          classifier(ch4, nclasses)...)                       # Classifier
end

end
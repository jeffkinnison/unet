from shadho import Shadho, spaces

# Search over some of the built-in optimizers
initializers = spaces.choice(['zeros', 'ones', 'identity', 'glorot_normal',
                              'glorot_uniform', 'he_normal', 'he_uniform',
                              'random_normal', 'random_uniform', ])

# Search over some of the built-in regularizers
regularizers = spaces.choice([None, 'l1', 'l2', 'l1_l2'])

# Search over some of the built-in constants
constraints = spaces.choice([None, 'non_neg', 'unit_norm'])

# Search over the built-in activations, parameterizing where necessary
activations = spaces.scope(
    exclusive=True,
    elu='elu',
    hard_sigmoid='hard_sigmoid',
    leaky_relu=spaces.scope('alpha': spaces.uniform(0, 1)),
    prelu=spaces.scope(
        alpha_initializer=initializers,
        alpha_regularizer=regularizers,
        alpha_constraints=constraints),
    relu='relu',
    sigmoid='sigmoid',
    softmax='softmax',
    softplus='softplus',
    softsign='softsign',
    tanh='tanh',
    thresholded_relu=spaces.scope(theta=spaces.uniform(-1, 1)))

# Set up a standard convolutional block that will search over all params that
# can be tuned for U-Net
conv = spaces.scope(
    kernel_size=spaces.randint(1, 12, 2),
    activation=activations,
    kernel_initializer=initializers,
    bias_initializer=initializers,
    kernel_regularizer=regularizers,
    bias_regularizer=regularizers,
    activity_regularizer=regularizers,
    kernel_constrains=constraints,
    bias_constraint=constraints)

# Search over the built-in optimizers, parameterizing SGD
optimizers = spaces.scope(
    exclusive=True
    sgd=spaces.scope(
        lr=spaces.log10_uniform(-4, -1),
        momentum=spaces.uniform(0, 1),
        decay=spaces.log10_uniform(-4, -1)),
    rmsprop='rmsprop',
    adagrad='adagrad',
    adadelta='adadelta',
    adam='adam',
    adamax='adamax',
    nadam='nadam')

# Set up the full search space over the U-Net down- and upsampling blocks
space = spaces.scope(
    optimizer=optimizers,
    min_filters=spaces.log2_randint(5, 8),
    down1=spaces.scope(conv1=conv, conv2=conv),
    down2=spaces.scope(conv1=conv, conv2=conv),
    down3=spaces.scope(conv1=conv, conv2=conv),
    down4=spaces.scope(conv1=conv, conv2=conv),
    up1=spaces.scope(conv1=conv, conv2=conv),
    up2=spaces.scope(conv1=conv, conv2=conv),
    up3=spaces.scope(conv1=conv, conv2=conv),
    up4=spaces.scope(conv1=conv, conv2=conv),
    out=spaces.scope(conv1=conv, conv2=conv))


if __name__ == '__main__':
    opt = Shadho()

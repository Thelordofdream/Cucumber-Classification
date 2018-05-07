import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


class CNN(object):
    def __init__(self, name="X", depth=9, input_size=32, batch_size=500, classes=10, residual_units=None,
                 learning_rate=0.01, decay_rate=0.0002, relu_leakiness=0.1, use_bottleneck=True):
        if residual_units is None:
            residual_units = [7, 7, 7]
        self.name = name
        self.depth = depth
        self.input_size = input_size
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate
        self.weight_decay_rate = decay_rate
        self.relu_leakiness = relu_leakiness
        self.use_bottleneck = use_bottleneck
        self.num_residual_units = residual_units
        self.mode = 'train'
        self.global_step = tf.train.get_or_create_global_step()

        self.x = None
        self.y = None
        self._extra_train_ops = []
        self.output = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.merged = None

    def create_nerual_network(self):
        with tf.variable_scope('init'):
            self.x = tf.placeholder("float", [self.batch_size, self.input_size, self.input_size, self.depth], name="x")
            # Convolution Layer 1（9,3x3/1,16）
            x = self._conv('init_conv', self.x, 3, self.depth, 16, self._stride_arr(1))

        # Strides for 3 ResNet units
        strides = [1, 2, 2]
        # Switch of activation for 3 ResNet units
        activate_before_residual = [True, False, False]
        if self.use_bottleneck:
            # bottleneck residual units
            res_func = self._bottleneck_residual
            # channel combination
            filters = [16, 64, 128, 256]
        else:
            # standard residual units
            res_func = self._residual
            # channel combination
            filters = [16, 16, 32, 64]

        # First group
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.num_residual_units[0]):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        # Second group
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2],
                         self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, self.num_residual_units[1]):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        # Third group
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, self.num_residual_units[2]):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        # global average pooling
        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.relu_leakiness)
            x = self._global_avg_pool(x)

        # fully connection layer + SoftMax layer
        with tf.variable_scope('logit'):
            self.output = tf.nn.softmax(self._fully_connected(x, self.classes))

        # loss function
        with tf.variable_scope('costs'):
            self.y = tf.placeholder("float", [self.batch_size, self.classes], name="y")
            # average cross entropy
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.y), name='cross_entropy')
            # L2 regularization, weight decay
            self.loss = cross_entropy + self._decay()

        tf.summary.scalar('loss', self.loss)

        with tf.name_scope('optimizer'):
            # use AdamOptimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # merge the optimizer with extra updating operations
            train_ops = [optimizer] + self._extra_train_ops
            self.optimizer = tf.group(*train_ops)

            # compute accuracy
            correct_pred = tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    # Transfer stride number to stride array
    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

    # standard residual unit
    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        # Whether put activation before the residual layer
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                # batch normalization and ReLU activation first
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)
                # get identity
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                # get identity
                orig_x = x
                # batch normalization and ReLU activation
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)

        # First sub-layer
        with tf.variable_scope('sub1'):
            # 3x3 convolution，channels fit: in_filter -> out_filter
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        # Second sub-layer
        with tf.variable_scope('sub2'):
            # batch normalization and ReLU activation
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.relu_leakiness)
            # 3x3 convolution，1 stride，channels don't change: out_filter
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        # merge residual layer
        with tf.variable_scope('sub_add'):
            # when channels change
            if in_filter != out_filter:
                # average pooling with zero padding
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                # zero padding on channels
                orig_x = tf.pad(orig_x,
                                [[0, 0],
                                 [0, 0],
                                 [0, 0],
                                 [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                                 ])
            # merge identity and layer output
            x += orig_x

        # tf.logging.debug('image after unit %s', x.get_shape())
        return x

    # bottleneck residual unit
    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        # Whether put activation before the residual layer
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                # batch normalization and ReLU activation first
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)
                # get identity
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                # get identity
                orig_x = x
                # batch normalization and ReLU activation
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)

        # First sub-layer
        with tf.variable_scope('sub1'):
            # 1x1 convolution，channels fit: in_filter -> out_filter/4
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        # Second sub-layer
        with tf.variable_scope('sub2'):
            # batch normalization and ReLU activation
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.relu_leakiness)
            # 3x3 convolution，1 stride ，channels don't change: out_filter/4
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        # Third sub-layer
        with tf.variable_scope('sub3'):
            # batch normalization and ReLU activation
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.relu_leakiness)
            # 1x1 convolution，1 stride， channels don't change: out_filter/4 -> out_filter
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        # merge residual layer
        with tf.variable_scope('sub_add'):
            # when channels change
            if in_filter != out_filter:
                # 1x1 convolution，channels fit: in_filter -> out_filter
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)

            # merge identity and layer output
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # batch normalization
    # ((x-mean)/var)*gamma+beta
    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            # get the number of channels
            params_shape = [x.get_shape()[-1]]
            # offset
            beta = tf.get_variable('beta',
                                   params_shape,
                                   tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            # scale
            gamma = tf.get_variable('gamma',
                                    params_shape,
                                    tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                # comput mean and standard deviation on each channel
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                # create variables for batch mean and standard deviation
                moving_mean = tf.get_variable('moving_mean',
                                              params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance',
                                                  params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)
                # update batch mean and standard deviation by moving average
                # moving_mean = moving_mean * decay + mean * (1 - decay)
                # moving_variance = moving_variance * decay + variance * (1 - decay)
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                # get accumulated batch mean and standard deviation
                mean = tf.get_variable('moving_mean',
                                       params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32),
                                       trainable=False)
                variance = tf.get_variable('moving_variance',
                                           params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # batch normalization layer：((x-mean)/var)*gamma+beta
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    # weight decay, L2 regularization loss
    def _decay(self):
        costs = []
        # find a 'DW' variables
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        # get sum and multiply with weight decay rate
        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

    # 2D convolution layer
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            # create new convolution kernel and use random normal initializer
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            # build 2d convolution layer
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    # leaky ReLU function
    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    # fully connection layer
    def _fully_connected(self, x, out_dim):
        # shape the input to 2D tensor，size = [N,-1]
        x = tf.reshape(x, [self.batch_size, -1])
        # define w，use random uniform initializer，[-sqrt(3/dim), sqrt(3/dim)]*factor
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.initializers.variance_scaling(distribution="uniform"))
        # define b，use zero initializer
        b = tf.get_variable('biases', [out_dim], initializer=tf.zeros_initializer())
        # compute x*w+b
        return tf.nn.xw_plus_b(x, w, b)

    # global average pooling
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def load_data_source(self, filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([1], tf.int64),
                                               'image': tf.FixedLenFeature([], tf.string),
                                           })
        self.images = tf.decode_raw(features['image'], tf.int32)
        self.images = tf.reshape(self.images, [self.depth, self.input_size, self.input_size])
        self.images = tf.transpose(self.images, [1, 2, 0])
        self.label = tf.cast(features['label'], tf.int32)

        x_batch, y_batch = tf.train.shuffle_batch([self.images, self.label], batch_size=self.batch_size,
                                                  capacity=1000, min_after_dequeue=200, num_threads=4)
        index = tf.expand_dims(tf.range(0, self.batch_size), 1)
        concated = tf.concat([index, y_batch], axis=1)
        y_batch = tf.cast(tf.sparse_to_dense(concated, [self.batch_size, self.classes], 1.0, 0.0), dtype=tf.float32)

        if self.depth == 3:
            tf.summary.image('images', tf.cast(x_batch, dtype=tf.uint8))

        return x_batch, y_batch

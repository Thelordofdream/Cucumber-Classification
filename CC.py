import tensorflow as tf
import model
import matplotlib.pyplot as plt


def train(model, batch_x, batch_y, sess, training_iters=300, display_step=10):
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    sess.run(init)
    step = 1
    while step <= training_iters:
        batch_xs, batch_ys = sess.run([batch_x, batch_y])
        sess.run(model.optimizer, feed_dict={model.x: batch_xs, model.y: batch_ys})
        if step % display_step == 0:
            summary, loss, acc = sess.run([model.merged, model.loss, model.accuracy], feed_dict={model.x: batch_xs, model.y: batch_ys})
            train_writer.add_summary(summary, step)
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


def test(model, x_test, y_test, sess):
    test_data, test_label = sess.run([x_test, y_test])
    loss, acc = sess.run([ model.loss, model.accuracy], feed_dict={model.x: test_data, model.y: test_label})
    print("Test Loss= " + "{:.6f}".format(loss) + ", Test Accuracy= " + "{:.5f}".format(acc))

def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    # mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train_data = model.data(filename="train.tfrecord")
    test_data = model.data(filename="test.tfrecord")
    x_batch, y_batch = train_data.get_batches()
    x_test, y_test = test_data.get_batches()
    my_network = model.CNN(name="CC", learning_rate=0.001)
    my_network.create_nerual_network()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)
        print("===== Start Training! =====")
        train(my_network, x_batch, y_batch, sess)
        test(my_network, x_test, y_test, sess)
        coord.request_stop()
        coord.join(threads)
        save(sess)
import tensorflow as tf
import numpy as np
import model


def train(model, batch_x, batch_y, x_test, y_test, sess, training_iters=5000, test_step=10):
    train_writer = tf.summary.FileWriter('./train_v1', sess.graph)
    generation_error = []
    step = 1
    while step <= training_iters:
        batch_xs, batch_ys = sess.run([batch_x, batch_y])
        sess.run(model.optimizer, feed_dict={model.x: batch_xs, model.y: batch_ys})
        if step % test_step == 0:
            summary, loss, acc = sess.run([model.merged, model.loss, model.accuracy],
                                          feed_dict={model.x: batch_xs, model.y: batch_ys})
            train_writer.add_summary(summary, step)
            print("Iter " + str(step) + ":\nTrain Loss = " + "{:.6f}".format(
                loss) + ", Train Accuracy = " + "{:.5f}".format(acc))
            test_loss = test(model, x_test, y_test, sess)
            print("Gen Error = " + "{:.6f}".format(test_loss - loss))
            generation_error.append(test_loss - loss)
        step += 1
    print("Optimization Finished!")
    np.save('generation_v1.npy', np.array(generation_error))
    train_writer.close()


def test(model, x_test, y_test, sess):
    test_data, test_label = sess.run([x_test, y_test])
    loss, acc = sess.run([model.loss, model.accuracy], feed_dict={model.x: test_data, model.y: test_label})
    print("Test Loss = " + "{:.6f}".format(loss) + ", Test Accuracy = " + "{:.5f}".format(acc))
    return loss


def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model_v1/model.ckpt")
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    my_network = model.CNN(name="CC_v1", learning_rate=0.01, depth=3, classes=9)

    my_network.create_nerual_network()

    x_batch, y_batch = my_network.load_data_source(filename="./cc_v1/train.tfrecord")
    x_test, y_test = my_network.load_data_source(filename="./cc_v1/test.tfrecord")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)
        print("===== Start Training! =====")
        train(my_network, x_batch, y_batch, x_test, y_test, sess)
        coord.request_stop()
        coord.join(threads)
        save(sess)
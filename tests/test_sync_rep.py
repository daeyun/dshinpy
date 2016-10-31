import tempfile
import time

import tensorflow as tf


def main(_):
    sync = True
    start_chief_queue_runners = True

    is_chief = True
    server = tf.train.Server.create_local_server()
    logdir = tempfile.mkdtemp()

    graph = tf.Graph()

    device_setter = tf.train.replica_device_setter(worker_device='/job:worker/task:0')
    with graph.as_default(), tf.device(device_setter):
        # Build loss op.
        x = tf.random_normal([100, 128])
        y = tf.Variable(initial_value=tf.random_normal([100, 128]))
        global_step = tf.Variable(0)
        loss = tf.reduce_sum(tf.squared_difference(x, y), name='loss')

        # Set up optimizer.
        optim = tf.train.GradientDescentOptimizer(0.001)
        if sync:
            optim = tf.train.SyncReplicasOptimizerV2(optim, 1)
        minimize = optim.minimize(loss, global_step=global_step, name='train_op')

        ready_for_local_init = None
        local_step_init = None
        if sync:
            init_tokens = optim.get_init_tokens_op()
            chief_qr = optim.get_chief_queue_runner()
            ready_for_local_init = optim.ready_for_local_init_op
            local_step_init = optim.local_step_init_op
        init_op = tf.initialize_all_variables()

    sv = tf.train.Supervisor(graph=graph,
                             is_chief=is_chief,
                             ready_for_local_init_op=ready_for_local_init,
                             init_op=init_op,
                             local_init_op=local_step_init,
                             recovery_wait_secs=1,
                             global_step=global_step,
                             logdir=logdir)

    config = server.server_def.default_session_config
    sess = sv.prepare_or_wait_for_session(server.target, config=config,
                                          start_standard_services=False)

    with sess.as_default():
        if is_chief and sync and start_chief_queue_runners:
            sv.start_queue_runners(sess, [chief_qr])
            init_tokens.run()

    # Wait for the queue runner to start.
    time.sleep(2)

    tf.Session.reset(server.target)

    time.sleep(1)
    print('restarted session')

    server.join()


if __name__ == '__main__':
    tf.app.run(main)

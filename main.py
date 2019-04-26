import os, random
import tensorflow as tf

from model import classification_loss, vgg19, resnet152, inception_v4, senet154, densenet201 
import preprocess as pp

MAX_EPOCH = 150
BATCH_SIZE = 32
INPUT_SIZE = (224, 224)
GRAYSCALE = True
NUM_CLASS = 10

tf.app.flags.DEFINE_boolean("train", False, "Perform training or test. [False]")
tf.app.flags.DEFINE_string("logdir", "log", "Directory to store the training log. [log]")
tf.app.flags.DEFINE_string("datadir", "train_data", "Directory to store the training log. [log]")
settings = tf.app.flags.FLAGS
checkpoint_dir = os.path.join(settings.logdir, "checkpoint")
log_dir = settings.logdir

def load_im(file_name):
    with tf.variable_scope("load_image"):
        im = pp.to_float(pp.load_im(file_name))
    return im

def preprocess(im, augment=False):
    with tf.variable_scope("preprocess"):
        if augment:
            im = pp.random_resize(im, INPUT_SIZE)
            im = pp.random_flip(im)
            im = pp.random_distort_color(im)
            if GRAYSCALE: im = pp.grayscale(im)
            im = pp.random_op(im, [lambda _: _,
                pp.sharpen, pp.avg_blur, pp.unsharpen, pp.gauss_blur
            ])
        else:
            im = pp.resize(im, INPUT_SIZE)
            if GRAYSCALE: im = pp.grayscale(im)
        im = pp.normalize(im, 0.5, 0.5)
    return im

def build_net(trainable, x, drop_rate=0):
    with tf.variable_scope("label_map"):
        label_map = tf.Variable([""]*NUM_CLASS, dtype=tf.string, trainable=False, name="epoch")
    with tf.name_scope("VGG19"):
        logits = vgg19(trainable, x, NUM_CLASS)
    return logits, label_map

def build_optimizer(loss, global_step):
    lr = tf.train.exponential_decay(1e-4, global_step, 2000, 0.9)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, update_ops])
    return train_op, lr, optimizer

def load_train_set():
    dataset = pp.load_data(settings.datadir, exclude=["nonbullying"])
    nb_dataset = pp.load_data(settings.datadir, include=["nonbullying"])
    images, labels = [], []
    label_map = []
    print("Load Train Data:")
    for lab, im in dataset.items():
        images.extend(im) 
        labels.extend([len(label_map)]*len(im)) 
        label_map.append(lab)
        print("   ", lab, len(im))
    nb_images = random.choices(nb_dataset["nonbullying"], k=min(len(nb_dataset["nonbullying"]), int(1.5*len(images))))
    label_map.append("nonbullying")
    print("   ", "nonbullying", len(nb_images))
    print("Total", len(images)+len(nb_images))
    print("Label Map:")
    for i, name in enumerate(label_map):
        print("   ", name, i)

    with tf.name_scope("db"):
        data = (images + nb_images, labels+[label_map.index("nonbullying")]*len(nb_images))
        db = tf.data.Dataset.from_tensor_slices(data)
        db = db.map(lambda _, __: (load_im(_), __), num_parallel_calls=4)
        db = db.cache()
        db = db.map(lambda _, __: (preprocess(_, True), __), num_parallel_calls=4)
        db = db.shuffle(buffer_size=(len(images)+len(nb_images))).batch(BATCH_SIZE).prefetch(BATCH_SIZE*4)
        it = db.make_initializable_iterator()

    return label_map, db, it

def load_val_set(label_map=None):
    val_images, val_labels = [], []
    val_dataset = pp.load_data("test_data")
    new_label_map = [] if label_map is None else label_map
    print("Load Val Data:")
    for lab, im in val_dataset.items():
        val_images.extend(im)
        val_labels.extend([len(new_label_map) if label_map is None else label_map.index(lab)]*len(im))
        if label_map is None: new_label_map.append(lab)
        print("   ", lab, len(im))
    print("Total", len(val_images))
    if label_map is None:
        print("Label Map:")
        for i, name in enumerate(label_map):
            print("   ", name, i)

    with tf.name_scope("val_db"):
        val_db = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_db = val_db.map(lambda _, __: (preprocess(load_im(_)), __), num_parallel_calls=4)
        val_db = val_db.batch(BATCH_SIZE)
        val_db = val_db.cache()
        val_it = val_db.make_initializable_iterator()

    return new_label_map, val_db, val_it


def train():
    import datetime
    drop_rate = tf.placeholder_with_default(0.0, shape=(), name="drop_rate")
    
    label_map, train_db, train_it = load_train_set()
    label_map, val_db, val_it = load_val_set(label_map)
    assert(len(label_map) == NUM_CLASS)
    db_initializer = tf.group([train_it.initializer, val_it.initializer])

    with tf.name_scope("db_handle"):
        train_db_handle = train_it.string_handle()
        val_db_handle = val_it.string_handle()
        handle = tf.placeholder(tf.string, shape=[], name="db_handle")
        it_dummy = tf.data.Iterator.from_string_handle(handle, train_db.output_types, train_db.output_shapes)
        data, label = it_dummy.get_next()

    logits, label_map_tensor = build_net(True, data, drop_rate)
    _, acc, loss = classification_loss(data, logits, label)
    setup_label_map = tf.assign(label_map_tensor, label_map)

    with tf.name_scope("optimizer"):
        epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="epoch")
        inc_epoch = tf.assign_add(epoch, 1, name="inc_epoch")
        global_step = tf.train.get_or_create_global_step()
        train_op, lr, _ = build_optimizer(loss, global_step)

    summaries = [
        tf.summary.scalar("train/loss", loss),
        tf.summary.scalar("train/accuracy", acc),
        tf.summary.scalar("train/lr", lr),
    ]
    summary_op = tf.summary.merge(summaries)

    saver = tf.train.Saver(max_to_keep=10)
    with tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir) as sess:
        logger = tf.summary.FileWriter(
            os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
            sess.graph
        )
        epo, train_handle, val_handle, *_ = sess.run([
            epoch, train_db_handle, val_db_handle, db_initializer, setup_label_map
        ])

        while epo < MAX_EPOCH:
            sess.run(db_initializer)
            print("Epoch: {} #########################################################".format(epo+1))
            acc_records, loss_records = [], []
            while not sess.should_stop():
                try:
                    summary, l, a, s, *_ = sess.run(
                        [summary_op, loss, acc, global_step, train_op],
                        feed_dict={handle: train_handle, drop_rate: 0.5}
                    )
                    acc_records.append(a)
                    loss_records.append(l)
                    logger.add_summary(summary, s)
                    if s % 50 == 0:
                        print("Step: {}, Loss: {:4}, Acc: {}".format(s, l, a))
                except tf.errors.OutOfRangeError:
                    break
            print("Train Loss: {:.4f}".format(sum(loss_records)/len(loss_records)))
            print("Train Accuracy: {:.4f}".format(sum(acc_records)/len(acc_records)))

            acc_records, loss_records = [], []
            while not sess.should_stop():
                try:
                    l, a = sess.run(
                        [loss, acc],
                        feed_dict={handle: val_handle}
                    )
                    acc_records.append(a)
                    loss_records.append(l)
                except tf.errors.OutOfRangeError:
                    l = sum(loss_records)/len(loss_records)
                    a = sum(acc_records)/len(acc_records)
                    logger.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag="val/loss", simple_value=l),
                        tf.Summary.Value(tag="val/acc", simple_value=a),
                    ]), s)
                    epo = sess.run(inc_epoch)
                    logger.flush()
                    saver.save(sess.raw_session(), checkpoint_dir+"/cpt", s)
                    break
            print("Validation Loss: {:.4f}".format(l))
            print("Validation Accuracy: {:.4f}".format(a))


def test():
    import sys
    if len(sys.argv) < 2: exit()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    images = sys.argv[1:]

    db = tf.data.Dataset.from_tensor_slices(images)
    db = db.map(lambda _: preprocess(load_im(_)), num_parallel_calls=8)
    db = db.batch(BATCH_SIZE).prefetch(3*BATCH_SIZE)
    it = db.make_one_shot_iterator()
    data = it.get_next()

    logits, label_map_tensor = build_net(False, data)
    predict = tf.argmax(logits, axis=-1)

    with tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir) as sess:
        label_map = sess.run(label_map_tensor)
        label_map = [s.decode("utf-8") for s in label_map]
        assert(len(label_map) == NUM_CLASS)
        while True:
            try:
                for p in sess.run(predict):
                    print(label_map[p])
            except tf.errors.OutOfRangeError:
                break

if __name__ == "__main__":
    if settings.train:
        train()
    else:
        test()
        
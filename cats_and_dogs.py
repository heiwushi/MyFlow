import os
import numpy as np
from PIL import Image
import myflow as mf
from matplotlib import pyplot as plt
from sklearn import metrics as metrics
folder_path = 'kagglecatsanddogs/PetImages'

IMG_SIZE = 64
BATCH_SIZE = 100
hidden_size1 = 128
hidden_size2 = 64


class FCModel(object):
    def __init__(self, image_size, class_num):
        self.x = mf.placeholder(shape=[BATCH_SIZE, image_size, image_size], tensor_name="x")
        self.x_flatten = mf.reshape(self.x, [BATCH_SIZE, image_size * image_size],
                                    tensor_name="x_flatten") / 255.00 - 0.5
        self.y = mf.placeholder(shape=[BATCH_SIZE, class_num], tensor_name="y")

        self.v1 = mf.variable(init_value=1 - 2 * np.random.random([image_size * image_size, hidden_size1]),
                              tensor_name="v1")
        self.b1 = mf.variable(init_value=np.zeros([hidden_size1]), tensor_name="b1")
        self.l1 = mf.nn.add_bias(mf.matmul(self.x_flatten, self.v1), self.b1, tensor_name="logits1")
        self.h1 = mf.relu(self.l1, tensor_name='h1')
        self.v2 = mf.variable(init_value=1 - 2 * np.random.random([hidden_size1, hidden_size2]), tensor_name="v2")
        self.b2 = mf.variable(init_value=np.zeros([hidden_size2]), tensor_name="b2")
        self.l2 = mf.nn.add_bias(mf.matmul(self.h1, self.v2), self.b2, tensor_name="logits2")
        self.h2 = mf.relu(self.l2)
        self.v3 = mf.variable(init_value=1 - 2 * np.random.random([hidden_size2, class_num]), tensor_name="v3")
        self.b3 = mf.variable(init_value=np.zeros([class_num]), tensor_name="b3")
        self.logits = mf.nn.add_bias(mf.matmul(self.h2, self.v3), self.b3, tensor_name="logits3")
        self.y_pred = mf.sigmoid(self.logits)


def read_pics_dataset():
    train_input_dataset = []
    train_label_dataset = []
    test_input_dataset = []
    test_label_dataset = []
    total_num = len(os.listdir(os.path.join(folder_path, 'Cat'))) + len(os.listdir(os.path.join(folder_path, 'Dog')))
    count = 0
    for i, label_name in enumerate(['Cat', 'Dog']):
        for pic_file in os.listdir(os.path.join(folder_path, label_name)):
            if pic_file.endswith('.jpg'):
                image = Image.open(os.path.join(folder_path, label_name, pic_file))
                image = image.resize((IMG_SIZE, IMG_SIZE))
                image = image.convert("L")
                count += 1
                if count % 10 == 0:
                    test_input_dataset.append(np.asarray(image))
                    test_label_dataset.append([i])
                else:
                    train_input_dataset.append(np.asarray(image))
                    train_label_dataset.append([i])
                if count % 1000 == 0:
                    print("read dataset:", round(float(count) / total_num * 100.0), "%")

    print("read dataset: 100 %")
    train_input_dataset = np.asarray(train_input_dataset)
    train_label_dataset = np.asarray(train_label_dataset)
    test_input_dataset = np.asarray(test_input_dataset)
    test_label_dataset = np.asarray(test_label_dataset)
    return train_input_dataset, train_label_dataset, test_input_dataset, test_label_dataset


def get_batch(input_dataset, label_dataset, BATCH_SIZE):
    assert len(input_dataset) == len(label_dataset)
    dataset_size = len(input_dataset)
    random_inds = np.random.choice(dataset_size, BATCH_SIZE)
    return input_dataset[random_inds], label_dataset[random_inds]


def main():
    plotter = Plotter()
    train_input_dataset, \
    train_label_dataset, \
    test_input_dataset, \
    test_label_dataset = read_pics_dataset()
    with mf.Graph() as g:
        model = FCModel(image_size=IMG_SIZE, class_num=1)
        x = model.x
        y = model.y
        y_pred = model.y_pred
        loss = mf.losses.binary_cross_entropy(y, y_pred)
        optimizer = mf.optimizer.RMSProp(learn_rate=0.001)
        vars_gradients = optimizer.compute_gradient(loss)
        train_step = optimizer.apply_gradient(vars_gradients)
        with mf.Session() as sess:
            for i in range(1, 10001):
                train_input_batch, train_label_batch = get_batch(train_input_dataset, train_label_dataset, BATCH_SIZE)
                _, loss_val, y_pred_val= sess.run([train_step, loss, y_pred], feed_dict={x: train_input_batch, y: train_label_batch})
                y_pred_val = np.asarray(list(map(lambda item:1 if item[0]>0.5 else 0, y_pred_val)), np.int8)
                train_accuracy = metrics.accuracy_score(train_label_batch, y_pred_val)
                print("train:", i, loss_val, train_accuracy)
                if i % 100 == 0:
                    test_input_batch, test_label_batch = get_batch(train_input_dataset, train_label_dataset, BATCH_SIZE)
                    test_loss_val, test_y_pred_val = sess.run([loss, y_pred],
                                                              feed_dict={x: test_input_batch, y: test_label_batch})
                    test_y_pred_val = np.asarray(list(map(lambda item: 1 if item[0] > 0.5 else 0, test_y_pred_val)), np.int8)
                    test_accuracy = metrics.accuracy_score(test_label_batch, test_y_pred_val)
                    print("validate:", i, test_loss_val, test_accuracy)
                    plotter.plot(i, train_accuracy, i, test_accuracy)
    plotter.show()


class Plotter(object):

    def __init__(self):
        plt.figure(figsize=(8, 6), dpi=80)
        plt.ion()
        self.xs = []
        self.ys = []
        self.test_xs = []
        self.test_ys = []

    def plot(self, x, y, test_x, test_y):
        plt.cla()
        plt.title("curve")
        plt.xlabel("step")
        plt.ylabel("loss")

        plt.grid(True)
        self.xs.append(x)
        self.ys.append(y)
        self.test_xs.append(test_x)
        self.test_ys.append(test_y)
        plt.plot(self.xs, self.ys, label="train")
        plt.plot(self.test_xs, self.test_ys, label="validate")
        plt.legend(loc="upper left", shadow=True)
        plt.pause(0.001)

    def show(self):
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
import tensorflow as tf
import sys
import math

num_hidden = 100
horizon = 20
num_layers = 1
rate = 0.001

data = tf.placeholder(tf.float32, [None, horizon, 256])
target = tf.placeholder(tf.float32, [horizon, 256])
stacked_rnn = []
for i in range(num_layers):
  stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True))
cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
zero_state = cell.zero_state(1, tf.float32)
input_state = tf.placeholder(tf.float32, [num_layers, 2, 1, num_hidden])
l = tf.unstack(input_state, axis=0)
rnn_tuple_state = tuple(
      [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
      for idx in range(num_layers)]
)
output, state = tf.nn.dynamic_rnn(cell, data, initial_state=rnn_tuple_state, dtype=tf.float32)
output = tf.squeeze(tf.transpose(output, [1, 2, 0]))

weight = tf.Variable(tf.truncated_normal([num_hidden, 256]))
bias = tf.Variable(tf.constant(0.1, shape=[256]))
prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
loss = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
minimize = optimizer.apply_gradients(capped_gvs)
#minimize = optimizer.minimize(loss)

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

last_byte = 0
cross_entropy = 0
byte_count = 0
cur_state = sess.run(zero_state)
thres = 0.0001
input_arr = [[]]
target_arr = []
byte_arr = []
with open(sys.argv[1:][0], "rb") as f:
    while True:
        byte = f.read(1)
        if not byte:
            break
        cur_byte = ord(byte)
        byte_arr.append(cur_byte)
        new_input = [0]*256;
        new_input[last_byte] = 1
        input_arr[0].append(new_input)
        new_target = [0]*256
        new_target[cur_byte] = 1
        target_arr.append(new_target)
        if len(target_arr) == horizon:
          cur_state, pred, _ = sess.run([state, prediction, minimize], {data: input_arr, target: target_arr, input_state: cur_state})
          for i in range(len(pred)):
            total = 0
            for j in range(len(pred[i])):
              if pred[i][j] < thres:
                pred[i][j] = thres
              total += pred[i][j]
            pred[i] = [x / total for x in pred[i]]
            cross_entropy += math.log(pred[i][byte_arr[i]], 2)
          target_arr = []
          byte_arr = []
          input_arr = [[]]
        byte_count += 1
        if byte_count % 1000 == 0:
          s = repr(byte_count) + ": " + repr(-cross_entropy/byte_count)
          print s
        last_byte = cur_byte

cross_entropy = -cross_entropy / byte_count
s = "cross entropy: " + repr(cross_entropy)
print s
s = "output size: " + repr(int(round((cross_entropy/8)*byte_count)))
print s

import tensorflow as tf
import numpy as np
import random as random

xy = np.loadtxt('price_training.csv', delimiter = ',', dtype = np.float32)
xz = np.loadtxt('price_test.csv', delimiter = ',', dtype = np.float32)
xw = np.loadtxt('price_val.csv', delimiter = ',', dtype = np.float32)
x_training_set = xy[:,0:-1] #row
y_training_set = xy[:,[-1]]

x_validation_set = xw[:,0:-1] #row
y_validation_set = xw[:,[-1]]

x_test_set = xz[0:10,0:-1]
y_test_set = xz[0:10,[-1]]

nb_classes = 2

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])


W1 = tf.Variable(tf.random_normal([4, 4]), name='weight')

b1 = tf.Variable(tf.random_normal([1]), name='bias')

X1 = tf.matmul(X, W1)+b1


W2 = tf.Variable(tf.random_normal([4,nb_classes]), name='weight')

b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias')

X2 = tf.nn.relu(tf.matmul(X1,W2)+b2)


W3 = tf.Variable(tf.random_normal([nb_classes,nb_classes]), name='weight')

b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias')

X3 = tf.matmul(X2,W3)+b3



W4 = tf.Variable(tf.random_normal([nb_classes,nb_classes]), name='weight')

b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias')

X4 = tf.sigmoid(tf.matmul(X3,W4)+b4)



W5 = tf.Variable(tf.random_normal([nb_classes,1]), name='weight')

b5 = tf.Variable(tf.random_normal([1]), name='bias')



logits = tf.matmul(X4, W5) + b5

hypothesis = logits

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


prediction = tf.argmax(hypothesis, 1) #우리가 만든 모델에서 가장 높은 값을 넣는다

correct_prediction = tf.equal(prediction, tf.argmax(Y, 1)) #위에서 나온 값이랑 해당 셈플의 y값이랑 분류가 일치하나?

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #위의것을 활용해서 cast해서 그걸 accuarcy 정확도 라고 한

saver = tf.train.Saver()

min_loss = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(900): #in은 안에 머가 있느냐 라는 물음
        idx = random.randint(0, x_training_set.size-1)
        sess.run(optimizer, feed_dict={X: x_training_set[idx:idx+1,:], Y: y_training_set[idx:idx+1, :]}) #여기서 학습된 모델 w가 나옴
        #idx에 랜덤으로 저장된 값을 저장하고 그 인덱스값에 해당하는 공의 데이터sample을 가져옴
        if step % 100 == 0:
            loss_train,acc_train = sess.run([cost, accuracy], feed_dict = {
                X: x_training_set,Y: y_training_set})
            print("Step: {:5}\ttrain_Loss: {:.3f}\t train_Acc: {:.2%}\n".format(step,loss_train, acc_train))
            loss_val, acc_val = sess.run([cost, accuracy], feed_dict = {X:x_validation_set,Y:y_validation_set})
            print("\t \t val_loss : {:.3f}\t val_acc: {:.2%}".format(loss_val, acc_val))
            if min_loss > loss_val:
                min_loss = loss_val
                save_path = saver.save(sess, r"C:\Users\wldnd\Desktop\CAPSTONE\project\muchandise_price_AI\model.ckpt")
                print("save complite: %s"%save_path)
            
    saver.restore(sess, r"C:\Users\wldnd\Desktop\CAPSTONE\project\muchandise_price_AI\model.ckpt")#q불러오고 밑에 코드에서 확인
    check_val_cost, check_val_acc = sess.run([cost,accuracy], feed_dict = {X:x_validation_set,Y:y_validation_set})
    print("\t \t ch_val_Loss: {:.3f}\t ch_val_Acc: {:.2%}".format(check_val_cost,check_val_acc))
    
    print("Learning finished")

    for step in range(1):
        loss_test,acc_test = sess.run([cost, accuracy], feed_dict = {
                X: x_test_set, Y: y_test_set}) #그래서 여기서 validation으로 실험
        print("test_set:  test_Loss: {:.3f}\t test_Acc: {:.2%}".format(loss_test, acc_test))
        
    pred = sess.run(prediction, feed_dict={X: x_test_set})

    for p,y in zip(pred, y_test_set.flatten()):
        #zip은 리스트를 짝지어줌
        if (p == int(y)):
            print("[맞습니다] Prediction: {} True Y: {}".format(p, int(y)))
        else:
            print("[틀립니다] Prediction: {} True Y: {}".format(p, int(y)))
        #예측값과 그라운드 트루의 일치여부
   
    #for step in range(y_test_set.size):
        
      #  r = random.randint(0, y_test_set.size-1)
      #  print("Label: ", sess.run(tf.argmax(y_test_set[r:r + 1], 1)))
      #  print("Prediction: ", sess.run(
      #      tf.argmax(hypothesis, 1), feed_dict={X: x_test_set[r:r + 1]}))


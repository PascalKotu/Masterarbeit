import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
import PIL
import time
from tensorflow.keras import layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pathlib
import tensorflow_probability as tfp
from sklearn.metrics import brier_score_loss
from code_utils import SaveResults, MakeModel, ImportImages


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(tf.__version__)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 32
img_height = 180
img_width = 180
alpha = 0.75
lam_u = 1.5
epochs = 200

train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)



#import unlabeled images
unlabeled_Images = []
data_dir = pathlib.Path('E:/Masterarbeit/unlabeled_data')
print("Importing images from:", data_dir)

image_Paths = list(data_dir.glob('*'))
print(len(image_Paths), "images found")

for i in image_Paths:
	image = PIL.Image.open(i).convert('RGB').resize((img_width, img_height))
	image = np.asarray(image)

	unlabeled_Images.append(image)

unlabeled_Images = np.asarray(unlabeled_Images)	
unlabeled_Images = (unlabeled_Images - 127.5) / 127.5



for run in range(5):

    #convert images into keras datasets
    tf.keras.backend.clear_session()
	
    train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels))
    train_ds = train_ds.batch(batch_size)


    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(batch_size)
	

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(4000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    

    #define data augmentation
    data_augmentation = tf.keras.Sequential(
		    [layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
		    layers.experimental.preprocessing.RandomRotation(0.1),
		    layers.experimental.preprocessing.RandomZoom(0.1),]
    )


    model = MakeModel(4)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)


    tf.config.run_functions_eagerly(True)

    def MixMatch(images, labels, images_unlabled):
        #create augmented images
        aug_images = data_augmentation(images, training=True)

        unlabled_augmented_images = []
        unlabled_guessed_lables = []

        #create augmented unlabeld images with k augmentations per image and guessed averaged shrpened labels
        for image_unlabeled in images_unlabled:
            augmented_image = data_augmentation( tf.expand_dims(image_unlabeled, axis = 0), training=True)
            averagePrediction = model([augmented_image], training=True)
            all_image_augmentations = augmented_image

            for k in range(1):
                augmented_image = data_augmentation( tf.expand_dims(image_unlabeled, axis = 0), training=True)
                averagePrediction += model([augmented_image], training=True)
                all_image_augmentations = tf.concat([all_image_augmentations, augmented_image], axis = 0)

            averagePrediction = averagePrediction/2

            mean = 0
            for pi in range(len(averagePrediction[0])):
                mean += averagePrediction[0,pi] ** (1/0.5)

            sharpenedCategories = []
            for pi in range(len(averagePrediction[0])):
                sharpenedCategories.append((averagePrediction[0,pi] ** (1/0.5)) / mean)

            sharpenedPrediction = tf.convert_to_tensor(sharpenedCategories, dtype=tf.float32)
            for i in all_image_augmentations:
                unlabled_augmented_images.append(i)
                unlabled_guessed_lables.append(sharpenedPrediction)

        unlabled_augmented_images = tf.convert_to_tensor(unlabled_augmented_images, dtype=tf.float32)
        unlabled_guessed_lables = tf.convert_to_tensor(unlabled_guessed_lables, dtype=tf.float32)
    
        #concatenate all augmented images and labels and shuffle them
        w_images = tf.concat([aug_images, unlabled_augmented_images], axis = 0)
        w_labels = tf.concat([labels, unlabled_guessed_lables], axis = 0)
        
        indices = tf.range(start=0, limit=tf.shape(w_images)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        w_images = tf.gather(w_images, shuffled_indices)
        w_labels = tf.gather(w_labels, shuffled_indices)

        

        #create x' and u' using mixup
        x_dash_images = []
        x_dash_labels = []
        for i in range(len(aug_images)):
            lam = np.random.beta(alpha, alpha)
            lam = np.max([lam, 1-lam])
            x_dash_images.append( lam*aug_images[i] + (1-lam)* w_images[i] )
            x_dash_labels.append( lam*labels[i] + (1-lam)* w_labels[i] )

        x_dash_images = tf.convert_to_tensor(x_dash_images, dtype=tf.float32)
        x_dash_labels = tf.convert_to_tensor(x_dash_labels, dtype=tf.float32)

        u_dash_images = []
        u_dash_labels = []
        for i in range(len(unlabled_augmented_images)):
            lam = np.random.beta(alpha, alpha)
            lam = np.max([lam, 1-lam])
            u_dash_images.append( lam*unlabled_augmented_images[i] + (1-lam)* w_images[i+len(aug_images)] )
            u_dash_labels.append( lam*unlabled_guessed_lables[i] + (1-lam)* w_labels[i+len(aug_images)] )

        u_dash_images = tf.convert_to_tensor(u_dash_images, dtype=tf.float32)
        u_dash_labels = tf.convert_to_tensor(u_dash_labels, dtype=tf.float32)

        return x_dash_images, x_dash_labels, u_dash_images, u_dash_labels


    @tf.function
    def train_step(x_dash_images, x_dash_labels, u_dash_images, u_dash_labels, epoch):
        #applay gradients based on x' and u'
        loss = 0
        with tf.GradientTape() as model_tape:

            x_output = model(x_dash_images, training=True)
            labeled_loss = cce(x_dash_labels, x_output)

            u_output = model(u_dash_images, training=True)
            unlabled_loss = cce(u_dash_labels,u_output)
            unlabled_loss = lam_u * unlabled_loss

            if epoch < 0:
                loss = labeled_loss
            else:
                loss = labeled_loss +  unlabled_loss


        gradients_of_model = model_tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        return loss, labeled_loss, unlabled_loss




    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Start training...")


    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print("Epoch:",epoch)
        start = time.time()

        step = 0
        loss = 0
        loss_labeled = 0
        loss_unlabeled = 0
        for image_batch, label_batch in train_ds:
            #unlabled_ds = unlabled_ds.shuffle(1000)
            #unlabeled_batch = unlabled_ds.take(batch_size)
            np.random.shuffle(unlabeled_Images)
            unlabeled_batch = unlabeled_Images[0:batch_size]
            x_dash_images, x_dash_labels, u_dash_images, u_dash_labels = MixMatch(image_batch, label_batch, unlabeled_batch)
                
            batch_loss, batch_loss_labeled, batch_loss_unlabeled = train_step(x_dash_images, x_dash_labels, u_dash_images, u_dash_labels, epoch)
            loss += batch_loss
            loss_labeled += batch_loss_labeled
            loss_unlabeled += batch_loss_unlabeled

            logits = model(image_batch, training=False)
            train_acc_metric.update_state(label_batch, logits)
            print("\tStep:",str(step+1),"of",len(train_ds),"Training acc: %.4f" % (float(train_acc_metric.result()),),"Training loss: %.4f" % (batch_loss,), "labeled loss: %.4f" % (batch_loss_labeled,), "unlabeled loss: %.4f" % (batch_loss_unlabeled,), end='\x1b\r')
            step+= 1

        
        loss = loss/len(train_ds)
        loss_unlabeled = loss_unlabeled/len(train_ds)
        loss_labeled = loss_labeled/len(train_ds)
        print()



        # Reset training metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()
        val_loss = 0
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_ds:
            val_logits = model(x_batch_val, training=False)
            val_loss += cce(y_batch_val, val_logits)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_loss = val_loss/len(val_ds)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        train_losses.append([loss, loss_labeled, loss_unlabeled])
        val_losses.append(val_loss)
        print("\tTraining acc over epoch: %.4f" % (float(train_acc),), "Training loss: %.4f" % (loss,),"labeled loss: %.4f" % (loss_labeled,), "unlabeled loss: %.4f" % (loss_unlabeled,), "Validation acc: %.4f" % (float(val_acc),), "Validation loss: %.4f" % (float(val_loss),))


        print ('\tTime for epoch {} is {} sec'.format(epoch, time.time()-start))



    print("Training finished...")


    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    test_loss = 0
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in test_ds:
        val_logits = model(x_batch_val, training=False)
        test_loss += cce(y_batch_val, val_logits)
        # Update val metrics
        test_acc_metric.update_state(y_batch_val, val_logits)

    test_loss = test_loss/len(test_ds)
    results = [test_acc_metric.result(), test_loss]
    test_acc_metric.reset_states()


    print("test loss, test acc:", results)

    folder_name = "mixmatch_results"
    if os.path.exists('models\\'+folder_name+'\\'+str(run)) == False:
	    os.makedirs('models\\'+folder_name+'\\'+str(run))
    model.save('models/'+folder_name+'/'+str(run)+'/'+folder_name+'_3.h5')
    np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Acc_3.txt', np.asarray(train_accs))
    np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_acc_3.txt', np.asarray(val_accs))
    np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Loss_3.txt', np.asarray(train_losses))
    np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_loss_3.txt', np.asarray(val_losses))
    np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Results_3.txt', np.asarray([results]))

  
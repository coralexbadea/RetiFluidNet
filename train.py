import warnings
warnings.filterwarnings("ignore") #ignore all warnings (so peacefull)
from tqdm import tqdm
import numpy as np
from silence_tensorflow import silence_tensorflow #no warnings (lol they have that)
silence_tensorflow()# call it
import tensorflow as tf # tensorflow
tf.random.set_seed(12345) # set random 
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import KFold #Kfold 
from DataReader import DataReader #importing our data reader
from model import RetiFluidNet # the model
from losses import Losses, IntervalEvaluation #losses and evaluation
from results import Results # results 
import glob

dataset_name = 'Spectralis'  #Spectralis # Cirrus #Topcon

path = "/content/Images"

print("Dataset: {}".format(dataset_name))

data_path = [] #paths to images 
for path in glob.glob(path + '/*'):
    print("for paths: ", path)
    data_path.append(path)    
print("Number of cases : ", len(data_path))

data_reader = DataReader() # datareader
retiFluidNet = RetiFluidNet(input_shape=(256, 256, 1)) #retifluidnet class that will spit a model when called
loss_funcs = Losses() #loses
my_results = Results() #results


train_falg = 1 # training
do_continue = False # continue
last_epoch = 20 #last_epoch

SEED = 100
NUM_EPOCHS = 1
BATCH_SIZE = 1#*nb_GPUs
BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE

kf = KFold(n_splits = 3, shuffle=False) # the kfold into 3
i = 3
overall_results = [] # overall results


def decay_schedule(epoch, lr): # lr decay schedule
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.8
    return lr


for train_path, val_path in kf.split(data_path): # train_path, val path
    tf.random.set_seed(12345)
    if i<=3: # to it 3 times
    
        print("Starting Fold number {}".format(i)) # we are at fold no i
        train_path, val_path = data_reader.get_trainPath_and_valPath(train_path, val_path, data_path) # get train val paths
        train_data, val_data = data_reader.get_data_for_train(train_path, val_path) #get train_data and val_data. train_data[0] spits image and mask
        num_of_train_samples = len(train_data)
        num_of_val_samples = len(val_data)
        for image, mask in val_data.skip(5).take(1):
            print("Image Shape : ", image.shape)
            print("Mask Shape  : ", mask.shape)
            test_image = image
            test_mask = mask

        # print("Starting Fold number {}".format(i))
        train_data = train_data.shuffle(buffer_size=BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE) #batch and prefetch using AUTOTUNE
        val_data = val_data.batch(1).prefetch(buffer_size = AUTOTUNE) #BZ of 1 and AUTOTUNE prefetch
        
        # with strategy.scope():
        model = retiFluidNet() # model of retifluidnet
        model.summary() #summary
        #model(tf.random.uniform(shape=[1,256,256,3]))
        initial_learning_rate = 2e-4 #lr
        decay_steps = 10000 #decay steps but we are not using this
        decay_rate  = 0.98 # decay rate but we are not using this one
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_schedule) # lr schedule

        
        if not os.path.exists(dataset_name): # exists dataset
            os.mkdir(dataset_name) #mkdir datasetname to save model and stuff
        
        # Creating Callbacks
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(dataset_name+"/model_%s_checkpoint.hdf5"%i,save_best_only=True) #checkpoint to save model
                                                            
        
        model.compile(optimizer = tf.keras.optimizers.RMSprop(initial_learning_rate), #RMS prop
                      loss = loss_funcs.training_loss, # training loss from loss funcs
                       metrics = [loss_funcs.dice]) #the dice loss from loss funcs
        
        if train_falg:
            ival = IntervalEvaluation(validation_data=val_data)
            if do_continue == True:
                model = tf.keras.models.load_model(dataset_name+"/model_%s_epoch%s.hdf5"%(i,last_epoch), custom_objects={'training_loss': loss_funcs.training_loss,
                                                                                                                             'dice_loss': loss_funcs.training_loss,
                                                                                                                             "dice":loss_funcs.dice})
                print("Pre-trained model loaded.")
            history = model.fit(train_data,
                                epochs=NUM_EPOCHS,
                                callbacks=[ival, lr_scheduler])
            model.save(dataset_name+"/model_%s_epoch%s.hdf5"%(i,NUM_EPOCHS))
            with open(dataset_name+"/model_%s_history.npy"%i, 'wb') as f:
                np.save(f, history.history)
        else:
            model = tf.keras.models.load_model(dataset_name+"/model_%s_epoch%s.hdf5"%(i,NUM_EPOCHS), custom_objects={'training_loss': loss_funcs.training_loss,
                                                                                                                             'dice_loss': loss_funcs.training_loss,
                                                                                                                             "dice":loss_funcs.dice})
            try:
                with open(dataset_name+"/model_%s_history.npy"%i, 'rb') as f:
                    History = np.load(f, allow_pickle=True).item()
            except:
                print("No history file is found.") 
        
        # plot learning curves
        if train_falg:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Learning Curves')
            
            axs[0].set_title('Model Loss')
            axs[0].plot(history.history['loss'], label='train')
            # axs[0].plot(history.history['val_loss'], label='val')
            axs[0].legend()
            axs[0].set(xlabel='Epoch', ylabel='Overall-Loss')
            
            axs[1].set_title('Model Dice Performance')
            axs[1].plot(history.history['main_output_dice_coeff'], label='train')
            axs[1].plot(history.history['val_main_output_dice_coeff'], label='val')
            axs[1].legend() 
            axs[1].set(xlabel='Epoch', ylabel='Main output dice_coeff')
            
            plt.show()
            fig.savefig(dataset_name+"\model_%s_history.png"%i, dpi=300)
        
        
        val_data = val_data.take(32)
        
        predictions = []
        for image, mask in tqdm(val_data):  
            temp = model.predict(image)[:, :, :, 0:32]
            predictions.append(temp)
        print("predictions: " , predictions)
        acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean = my_results.results_per_layer(predictions, val_data)
        overall_results.append([acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean])
    

        
        print('-'*50)
        print('Fold number {} finished'.format(i))
        print('-'*50)
        print('\n')
        print('\n')
        print('\n')
        print('\n')
    
 
        del model, train_data, val_data

    i -= 1     
    # break


# In[]:
my_results.print_overall_results(overall_results, dataset_name) 

print("SEED = %d\nNUM_EPOCHS = %d\nBATCH_SIZE = %d\nBUFFER_SIZE = %d"%(SEED,NUM_EPOCHS,BATCH_SIZE,BUFFER_SIZE))
print("initial_learning_rate = %.4f\ndecay_steps = %d\ndecay_rate = %0.2f"%(initial_learning_rate,decay_steps,decay_rate))
# In[]: END
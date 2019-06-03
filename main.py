#from model import *
#from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.01,
                    width_shift_range=0.005,
                    height_shift_range=0.005,
                    shear_range=0.005,
                    zoom_range=0.005,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(8,'data','masks','cnnmasks',data_gen_args,save_to_dir = 'generatedvae')#,image_color_mode = "rgb")

#model = get_cnn_model(nchannels=3)

#model_checkpoint = ModelCheckpoint('plantSegmentation/cnn_plant.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.load_weights('plantSegmentation/cnn_plant.hdf5')
#model.fit_generator(myGene,steps_per_epoch=50,epochs=10,callbacks=[model_checkpoint])
#
#testGene = testGenerator("data/images/imgs",as_gray=False,target_size=(1024,1024))
#results = model.predict_generator(testGene,7,verbose=1)
#saveResult("test",results)
vae, models = get_vae_model(rows=256,cols=256,nchannels=1)
model_checkpoint = ModelCheckpoint('vae_plant.hdf5', monitor='loss',verbose=1, save_best_only=True)
vae.fit_generator(myGene,steps_per_epoch=10,epochs=5,callbacks=[model_checkpoint])
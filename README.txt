WDnCNN: ENHANCING A CNN BASED DENOISER BY SPATIAL AND SPECTRAL ANALYSIS

This is the training and testing code in Pytorch for the WDnCNN. The whole project contains the following files

WDnCNN---denoised_figs(folder)
      			      ---noisy.png
      			      ---denoised.png
      ---log(folder)
      		    ---log.txt
		    ---log_c.txt
      ---model(folder)
		      ---WDnCNN_model_gray
		      ---WDnCNN_model_color
      ---testset(folder)
			---BSD68
				---test001.png, test002.png, test003.png, etc.
			---CBSD68
				---3096.png, 12084.png, 14037.png, etc.
			---Kodak24
				---kodim01.png, kodim02.png, kodim02.png, etc.
			---McMaster
				---01.tif, 02.tif, 03.tif, etc.
			---RNI6
				---Building.png, Chupa_Chups.png, David_Hilbert.png, etc.
			---RNI15
				---Audrey_Hepburn.jpg, Bears.png, Boy.png, etc.
			---Set12
      ---cfg.py
      ---data.py
      ---demo.py
      ---Loss_Function.py
      ---Network.py
      ---train.py
      ---utils.py


Test pretrained WDnCNN models:

Two pretrained WDnCNN models, WDnCNN_model_gray and WDnCNN_model_color, are used for testing. Run demo.py to test the WDnCNN for both synthetic and real-world noise removal. Due to the file size limit, we only uploaded tow real-world noisy databases: RNI6 (real-world grayscale), and RNI15(real-world color) for testing in this project. If you want to test WDnCNN on other datasets or images, you may download your dataset and change the file path in demo.py for correctly loading data. For synthetic noisy image restoration, a noise level sigma_n is needed to be set manually. The denoised images will be saved in 'denoised_figs' folder for further visual quality evaluation.

Train WDnCNN models:

To train the WDnCNN model, you need first download the three public databases: BSD400, Waterloo Exploration Database, and the validation set of ImageNet from their offical websites. Then modify the file path in data.py to correctly load the training data. You can change the 'mode' parameter in cfg.py to set the training mode. ('mode' can be 'gray' or 'color'. The 'color' mode is the default). Run the train.py to begin the training process.

To follow the band discriminative training criterion (BDT), the training process will end in every 50 epochs, and then you need to modify the weights mu_k of different bands in Loss_Function.py as shown in the paper. To resume training process, you need to set the correct 'beginner' in cfg.py, and load the last trained model by modifying the model name of 'net.load_state_dict()' in train.py. Run train.py again to continue training.

   Epoch   (LL, LH, HL, HH)     Epoch   (LL, LH, HL, HH)
  001-050(2.0, 2.5, 2.5, 4.5)  050-100(3.5, 2.5, 2.5, 3.0)
  100-150(4.5, 2.5, 2.5, 2.0)  150-200(5.5, 1.5, 1.5, 1.0)
  200-250(6.0, 2.0, 2.0, 1.5)  250-300(6.5, 2.5, 2.5, 2.0)
  300-350(7.0, 3.0, 3.0, 2.5)  350-400(7.5, 3.5, 3.5, 3.0)
  400-450(8.0, 4.0, 4.0, 3.5)  450-500(8.5, 4.5, 4.5, 4.0)

The training process will be recorded in ./log/log.txt or log_c.txt for gray and color mode, respectively. To use different wavelet filters for training, you can simply change the parameter 'wave_base' in cfg.py to sym2, sym8, db2, etc.

If you find any bug, please contact ruizhao19921010@gmail.com.

(This project may not be very well-organized, but it works. We will further revise it to be more structured and understandable.)

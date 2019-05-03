class par():
    bs = 128
    sigma = [0, 75]
    test_sigma = 25
    stride = 20
    aug_time = 1
    scales = [1, 0.9, 0.8, 0.7]
    mode = 'color'  # gray or color
    wave_base = 'dmey'
    patch_size = 50
    beginner = 0    # modify beginner to continue training from 'Epoch' beginner to Epoch 'beginner+50'
    if mode == 'gray':
        img_channel = 1
    else:
        img_channel = 3
    output_channel = 4 * img_channel
    input_channel = img_channel + 1
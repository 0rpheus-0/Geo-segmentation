import numpy as np
import matplotlib.pyplot as plt
from constants import CLASSES

def visualize_mask(mc_mask: np.ndarray):
    colors_imshow = {
        'City' : np.array([170, 240, 209]),
        'Cloud' : np.array([184, 61, 245]),
        'Field' : np.array([242, 242, 13]),
        'Mountains' : np.array([61, 61, 245]),
        'Sand': np.array([241, 184, 22]),
        'Trees': np.array([63, 220, 32]),
        'Water': np.array([13, 160, 236])
    }

    sc_mask = np.zeros((mc_mask[0].shape[0], mc_mask[0].shape[1], 3), dtype=np.uint8)
    square_ratios = {}

    for i, singlechannel_mask in enumerate(mc_mask):

        cls = CLASSES[i]
        singlechannel_mask = singlechannel_mask.squeeze()

        square_ratios[cls] = singlechannel_mask.sum() / singlechannel_mask.size
        
        sc_mask += np.multiply.outer(singlechannel_mask > 0, colors_imshow[cls]).astype(np.uint8)
        

    title = 'Square: ' + '\n'.join([f'{cls}: {square_ratios[cls]*100:.1f}%' for cls in CLASSES])
    return sc_mask, title

def visualize_multichennel_mask(img: np.ndarray, multichennel_mask: np.ndarray):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    img = img.transpose(2, 0, 1)
    axes[0].imshow(img[0], cmap="grey")
    multichennel_mask = multichennel_mask.transpose(2, 0, 1)
    mask_to_show, title = visualize_mask(multichennel_mask)
    axes[1].imshow(mask_to_show)
    axes[1].set_title(title)
    plt.tight_layout()
    plt.show()  
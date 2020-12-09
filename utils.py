import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_heatmap(fname, image, heatmap):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].set_title('original image')
    axes[0].axis('off')
    im1 = axes[0].imshow(image.squeeze(2), cmap='gray')
    
    axes[1].set_title('heatmap')
    axes[1].axis('off')
    im2 = axes[1].imshow(heatmap.squeeze(2), cmap='Reds')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    fig.show()

    if not os.path.exists('./results'):
        os.makedirs('./results')
    fig.savefig('./results/{}'.format(fname))


def visualize_heatmap_all(image, heatmap_dtd, heatmap_integrated):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].set_title('original image')
    axes[0].axis('off')
    im1 = axes[0].imshow(image.squeeze(2), cmap='gray')
    
    axes[1].set_title('Deep Taylor Decomposition')
    axes[1].axis('off')
    im2 = axes[1].imshow(heatmap_dtd.squeeze(2), cmap='Reds')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    axes[2].set_title('Integrated Gradients')
    axes[2].axis('off')
    im3 = axes[2].imshow(heatmap_integrated.squeeze(2), cmap='Reds')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    fig.show()
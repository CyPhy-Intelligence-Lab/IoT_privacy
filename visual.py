import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image



def save_gan(model, ep, **kwargs):
    name = model.__class__.__name__.lower()

    if name == "gan":
        data = model.call(5, training=False).numpy()
        plt.plot(data.T)
        plt.xticks((), ())
        dir_ = "visual/{}".format(name)
        os.makedirs(dir_, exist_ok=True)
        path = dir_ + "/{}.png".format(ep)
        plt.savefig(path)
    else:
        print("model can not be saved")
    plt.clf()
    plt.close()


def cvt_gif(folders_or_gan, shrink=10):
    if not isinstance(folders_or_gan, list):
        folders_or_gan = [folders_or_gan.__class__.__name__.lower()]
    for folder in folders_or_gan:
        folder = "visual/"+folder
        fs = [folder+"/" + f for f in os.listdir(folder)]
        imgs = []
        for f in sorted(fs, key=os.path.getmtime):
            if not f.endswith(".png"):
                continue
            try:
                int(os.path.basename(f).split(".")[0])
            except ValueError:
                continue
            img = Image.open(f)
            img = img.resize((img.width//shrink, img.height//shrink), Image.ANTIALIAS)
            imgs.append(img)
        path = "{}/generating.gif".format(folder)
        if os.path.exists(path):
            os.remove(path)
        imgs[-1].save(path, append_images=imgs, optimize=False, save_all=True, duration=400, loop=0)
        print("saved ", path)
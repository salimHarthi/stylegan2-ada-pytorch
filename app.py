import numpy as np
import PIL.Image
# import matplotlib.pyplot as plt
import torch
import legacy
import random 
import dnnlib
import streamlit as st
import gdown
from os.path import exists
st.header("Anime Face Generator")
seed = st.text_input('Enter a name', '')
submit = st.button('Generate')


if submit:
    if len(seed)>0 :
        with st.spinner('Wait for it...'):
            def seed2vec(G, seed):
                random.seed(seed)
                seed = random.randint(1, 1000)
                return np.random.RandomState(seed).randn(1, G.z_dim)

            # def display_image(image):
            #   plt.axis('off')
            #   plt.imshow(image)
            #   plt.show()


            def get_label(G, device, class_idx):
                label = torch.zeros([1, G.c_dim], device=device)
                if G.c_dim != 0:
                    if class_idx is None:
                        ctx.fail('Must specify class label with --class when using a conditional network')
                    label[:, class_idx] = 1
                else:
                    if class_idx is not None:
                        print ('warn: --class=lbl ignored when running on an unconditional network')
                return label

            def generate_image(device, G, z, truncation_psi=1.0, noise_mode='const', class_idx=None):
                z = torch.from_numpy(z).to(device)
                label = get_label(G, device, class_idx)
                print(label)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

            device = torch.device('cpu')
            URL = "https://drive.google.com/uc?id=10Fv4CrCCgietdhI8UqE-Ux0Q4zDL-mBN&export=download"
            output = 'model.pkl'
            file_exists = exists('model.pkl')
            if not file_exists:
                gdown.download(URL, output, quiet=False) 
            with open('model.pkl',"rb") as f:
            # with dnnlib.util.open_url(URL) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

            import functools
            G.forward = functools.partial(G.forward, force_fp32=True)
            z = seed2vec(G, seed)
            c = None
            img = generate_image(device, G, z)
            img.save("geeks.png")
            # display_image(img)
            st.image(img, caption='')

            with open("geeks.png", "rb") as file:

                btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name="face.png",
                        mime="image/png"
                    )
            st.success('Done!')
            

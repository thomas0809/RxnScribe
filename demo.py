import torch
from rxnscribe import MolDetect
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import imageio
import cv2
from PIL import Image
import numpy as np

from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoConfig

encoder_config = AutoConfig.from_pretrained('prajjwal1/bert-mini')
decoder_config = AutoConfig.from_pretrained('prajjwal1/bert-mini')
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=config)

print(model)



'''
ckpt_path = hf_hub_download("Ozymandias314/MolDetectCkpt", "coref_best.ckpt")
model = MolDetect(ckpt_path, coref = True)#, device=torch.device('cpu'))

model.predict_image_file("download-2.png", coref = True)


backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)



def convert_to_pil(image):
    if type(image) == np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return image

image_file = "download-2.png"
npimage = imageio.imread(image_file)
print(type(npimage)) 
predictions = model.predict_image(convert_to_pil(np.array(npimage.data)), coref = True)

print(predictions)

visualize_images = model.draw_bboxes(predictions, image_file = image_file, coref = True)

#plt.imshow(visualize_images[0])
#plt.show()

imageio.imwrite('output2-coref.png', visualize_images[0])
'''
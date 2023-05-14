import os
import torch
import yaml
from timm import create_model
from models import MemSeg
from data import create_dataset, create_dataloader
import matplotlib.pyplot as plt


cfg = yaml.load(open('./configs/capsule.yaml','r'), Loader=yaml.FullLoader)

# ====================================
# Select Model
# ====================================

def load_model(model_name):
    global model
    global testset 
    
    testset = create_dataset(
        datadir                = cfg['DATASET']['datadir'],
        target                 = model_name.split('-')[1], 
        train                  = False,
        resize                 = cfg['DATASET']['resize'],
        texture_source_dir     = cfg['DATASET']['texture_source_dir'],
        structure_grid_size    = cfg['DATASET']['structure_grid_size'],
        transparency_range     = cfg['DATASET']['transparency_range'],
        perlin_scale           = cfg['DATASET']['perlin_scale'], 
        min_perlin_scale       = cfg['DATASET']['min_perlin_scale'], 
        perlin_noise_threshold = cfg['DATASET']['perlin_noise_threshold']
    )
    
    memory_bank = torch.load(f'saved_model/{model_name}/memory_bank.pt', map_location=torch.device('cpu'))
    print(memory_bank)

    memory_bank.device = 'cpu'
    for k in memory_bank.memory_information.keys():
        memory_bank.memory_information[k] = memory_bank.memory_information[k].cpu()

    feature_extractor = feature_extractor = create_model(
        cfg['MODEL']['feature_extractor_name'], 
        pretrained    = True, 
        features_only = True
    )
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    )

    model.load_state_dict(torch.load(f'saved_model/{model_name}/best_model.pt', map_location=torch.device('cpu')))

# ====================================
# Visualization
# ====================================

def result_plot(idx, output_dir):
    input_i, mask_i, target_i = testset[idx]

    output_i = model(input_i.unsqueeze(0)).detach()
    output_i = torch.nn.functional.softmax(output_i, dim=1)

    def minmax_scaling(img):
        return (((img - img.min()) / (img.max() - img.min())) * 255).to(torch.uint8)

    fig, ax = plt.subplots(1,4, figsize=(15,10))
    
    ax[0].imshow(minmax_scaling(input_i.permute(1,2,0)))
    ax[0].set_title('Input: {}'.format('Normal' if target_i == 0 else 'Abnormal'))
    ax[1].imshow(mask_i, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[2].imshow(output_i[0][1], cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[3].imshow(minmax_scaling(input_i.permute(1,2,0)), alpha=1)
    ax[3].imshow(output_i[0][1], cmap='gray', alpha=0.5)
    ax[3].set_title(f'Input X Predicted Mask')
    
    plt.savefig(os.path.join(output_dir, f'image_{idx}.png'))
    plt.close()


# ====================================
# Model Selection and Output Directory
# ====================================

model_list = os.listdir('./saved_model')
model_name = 'MemSeg-capsule'

output_dir = './output_dir'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

load_model(model_name=model_name)

# ====================================
# Plotting and
# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Iterate over test set and save results
for idx in range(len(testset)):
    result_plot(idx, output_dir)

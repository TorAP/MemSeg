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
    global model1
    global model2
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
    
    
    memory_bank1 = torch.load(f'saved_model/{model_name}/memory_bank.pt', map_location=torch.device('cpu'))

    memory_bank1.device = 'cpu'
    for k in memory_bank1.memory_information.keys():
        memory_bank1.memory_information[k] = memory_bank1.memory_information[k].cpu()

    feature_extractor1 = feature_extractor1 = create_model(
        cfg['MODEL']['feature_extractor_name'], 
        pretrained    = True, 
        features_only = True
    )
    model1 = MemSeg(
        memory_bank       = memory_bank1,
        feature_extractor = feature_extractor1
    )

    model1.load_state_dict(torch.load(f'saved_model/{model_name}/{specific_model}', map_location=torch.device('cpu')))

    memory_bank2 = torch.load(f'saved_model/{model_name}/memory_bank.pt', map_location=torch.device('cpu'))

    memory_bank2.device = 'cpu'
    for k in memory_bank2.memory_information.keys():
        memory_bank2.memory_information[k] = memory_bank2.memory_information[k].cpu()

    feature_extractor2 = feature_extractor2 = create_model(
        cfg['MODEL']['feature_extractor_name'], 
        pretrained    = True, 
        features_only = True
    )
    model2 = MemSeg(
        memory_bank       = memory_bank2,
        feature_extractor = feature_extractor2
    )

    model2.load_state_dict(torch.load(f'saved_model/{model_name}/best_model.pt', map_location=torch.device('cpu')))
# ====================================
# Visualization
# ====================================

def result_plot(idx, output_dir):
    input_i, mask_i, target_i = testset[idx]

    output1_i = model1(input_i.unsqueeze(0)).detach()
    output1_i = torch.nn.functional.softmax(output1_i, dim=1)

    output2_i = model2(input_i.unsqueeze(0)).detach()
    output2_i = torch.nn.functional.softmax(output2_i, dim=1)
    def minmax_scaling(img):
        return (((img - img.min()) / (img.max() - img.min())) * 255).to(torch.uint8)

    fig, ax = plt.subplots(2,4, figsize=(12,8))
    ax[0][0].imshow(minmax_scaling(input_i.permute(1,2,0)))
    ax[0][0].set_title('Input: {}'.format('Normal' if target_i == 0 else 'Abnormal'))

    ax[0][1].imshow(mask_i, cmap='gray')
    ax[0][1].set_title('Ground Truth')

    ax[0][2].imshow(output1_i[0][1], cmap='gray')
    ax[0][2].set_title('Model 1 Predicted Mask')

    ax[0][3].imshow(minmax_scaling(input_i.permute(1,2,0)), alpha=1)
    ax[0][3].imshow(output1_i[0][1], cmap='gray', alpha=0.5)
    ax[0][3].set_title(f'Predicted Mask')

    ax[1][0].imshow(minmax_scaling(input_i.permute(1,2,0)))
    ax[1][0].set_title('Input: {}'.format('Normal' if target_i == 0 else 'Abnormal'))

    ax[1][1].imshow(mask_i, cmap='gray')
    ax[1][1].set_title('Ground Truth')

    ax[1][2].imshow(output2_i[0][1], cmap='gray')
    ax[1][2].set_title('Model 2 Predicted Mask')

    ax[1][3].imshow(minmax_scaling(input_i.permute(1,2,0)), alpha=1)
    ax[1][3].imshow(output2_i[0][1], cmap='gray', alpha=0.5)
    ax[1][3].set_title(f'Predicted Mask')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}.png'.format(idx)))
    plt.close()
   


# ====================================
# Model Selection and Output Directory
# ====================================

model_list = os.listdir('./saved_model')
model_name = 'MemSeg-capsule'
specific_model = "latest_model.pt"
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

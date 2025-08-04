print("train_fel_model.py started")
import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import joblib
from datetime import datetime
import warnings
import yaml
import argparse
warnings.filterwarnings("ignore")
from utils import parse_pv_yml
from lume_model.utils import variables_from_yaml, variables_as_yaml
from lume_model.models import TorchModel, TorchModule
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.utils import standardize
from botorch.models.transforms.input import AffineInputTransform, Normalize
import time



# File directory and pickle files
file_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset/'
pickle_files = ['hxr_archiver_2025-04.pkl',
                'hxr_archiver_2025-03.pkl','hxr_archiver_2025-02.pkl', 'hxr_archiver_2024-09.pkl',
                'hxr_archiver_2024-12.pkl', 'hxr_archiver_2024-11.pkl', 'hxr_archiver_2024-10.pkl']
                 # 'hxr_archiver_May_2024.pkl', 'hxr_archiver_Jun_2024.pkl', 'hxr_archiver_Sep_2024.pkl',
                # 'hxr_archiver_Oct_2024.pkl', 'hxr_archiver_Nov_2024.pkl'] #
print('Reading pickle files')
# Load and concatenate dataframes
dfs = [pd.read_pickle(file_dir+file) for file in pickle_files]
all_df = pd.concat(dfs, axis=0, ignore_index=False)

print('Number of total samples:', all_df.shape[0])

# Dataset filtering function
# wider beam and photon energy 
def dataset_filter(dataset):
    # Filtering based on multiple conditions
    condition = (dataset['ACCL:LI21:1:L1S_S_PV'] < 0) & (dataset['ACCL:LI21:1:L1S_S_AV'] > 100) & \
                (dataset['ACCL:LI22:1:ADES'] > 3000) &  (dataset['ACCL:LI22:1:ADES'] < 5400) & \
                (dataset['XRMS on VCC'] > 300) & (dataset['XRMS on VCC'] < 350) & \
                (dataset['YRMS on VCC'] > 250) & (dataset['YRMS on VCC'] < 350) & \
                (dataset['hxr_pulse_intensity'] > 0.02) & (dataset['hxr_pulse_intensity'] < 4.5) & \
               (dataset['Charge at gun [pC]'] > 240) & (dataset['Charge at gun [pC]'] < 260) & \
                (dataset['Charge after BC1 [pC]'] < 200) & \
                (dataset['HXR electron energy [GeV]'] > 8) & (dataset['HXR photon energy [eV]'] > 7000)
                # all_df['hxr_pulse_intensity'] > 0.05)
    return dataset[condition]
                # (dataset['Bunch length at BC1'] > 200) & (dataset['Bunch length at BC1'] < 300) & \
                # (dataset['Bunch length at BC2'] > 3e3) & (dataset['Bunch length at BC2'] < 1e5) & \


# def dataset_filter(dataset):
#     condition = (dataset['ACCL:LI21:1:L1S_S_PV'] > -40) & (dataset['ACCL:LI21:1:L1S_S_PV'] < 0) & \
#             (dataset['ACCL:LI21:1:L1S_S_AV'] > 100) & (dataset['ACCL:LI21:1:L1S_S_AV'] < 120) &\
#                 (dataset['ACCL:LI22:1:ADES'] > 3000) &  (dataset['ACCL:LI22:1:ADES'] < 5600) & \
#                 (dataset['ACCL:LI22:1:PDES'] > -45) &  (dataset['ACCL:LI22:1:PDES'] < -25) & \
#                 (dataset['XRMS on VCC'] > 280) & (dataset['XRMS on VCC'] < 360) & \
#                 (dataset['YRMS on VCC'] > 300) & (dataset['YRMS on VCC'] < 380) & \
#                 (dataset['hxr_pulse_intensity'] > 0.02) & (dataset['hxr_pulse_intensity'] < 4) & \
#                (dataset['Charge at gun [pC]'] > 240) & (dataset['Charge at gun [pC]'] < 260) & \
#                 (dataset['Bunch length at BC1'] > 180) & (dataset['Bunch length at BC1'] < 240) & \
#                  (dataset['Bunch length at BC2'] > 2500) & (dataset['Bunch length at BC2'] < 4700) & \
#                 (dataset['Charge after BC1 [pC]'] > 150) & (dataset['Charge after BC1 [pC]'] < 220) & \
#                 (dataset['HXR electron energy [GeV]'] > 8) & (dataset['HXR electron energy [GeV]'] < 14) & \
#                 (dataset['HXR photon energy [eV]'] > 5000) & (dataset['HXR photon energy [eV]'] > 20000)
#     return dataset[condition]

final_df = dataset_filter(all_df)
print('Number of total samples after filtering:', final_df.shape[0])
# Remove invalid quadrupoles
invalid_quad_list = []
for each in final_df.keys():
    if final_df[each].quantile(0) == final_df[each].quantile(1):
        invalid_quad_list.append(each)
        print(f'{each} should be removed')
print(f'Feature number is: {final_df.shape[1]}')
invalid_quad_list.extend(['QUAD:LI21:243:BCTRL', 'QUAD:LI24:713:BCTRL', 'QUAD:LI24:892:BCTRL',
                        'QUAD:CLTH:140:BCTRL', 'QUAD:CLTH:170:BCTRL', 'QUAD:BSYH:445:BCTRL',
                        'QUAD:LTUH:285:BCTRL', 'QUAD:LTUH:665:BCTRL', 'QUAD:DMPH:300:BCTRL',
                        'QUAD:DMPH:380:BCTRL', 'QUAD:DMPH:500:BCTRL', 'QUAD:BSYH:465:BCTRL',
                        'QUAD:BSYH:640:BCTRL', 'QUAD:BSYH:735:BCTRL', 'QUAD:BSYH:910:BCTRL',
                        'QUAD:LTUH:110:BCTRL', 'QUAD:LTUH:120:BCTRL', 'QUAD:LTUH:180:BCTRL',
                        'QUAD:LTUH:190:BCTRL', 'QUAD:LTUH:130:BCTRL', 'QUAD:LTUH:290:BCTRL',
                        'QUAD:LTUH:250:BCTRL', 'QUAD:LTUH:720:BCTRL', 'QUAD:LTUH:820:BCTRL',
                        'QUAD:UNDH:2780:BCTRL', 'QUAD:UNDH:2980:BCTRL', 'QUAD:UNDH:3080:BCTRL',
                        'QUAD:UNDH:4580:BCTRL', 'QUAD:UNDH:4680:BCTRL', 'QUAD:LI24:701:BCTRL',
                         'QUAD:LI24:601:BCTRL', 'QUAD:LI24:901:BCTRL', 'QUAD:LI25:201:BCTRL',
                         'QUAD:IN20:631:BCTRL', 'QUAD:IN20:651:BCTRL', 'QUAD:IN20:731:BCTRL',
                         'QUAD:LI21:315:BCTRL'])
# final_df = final_df.drop(columns=invalid_quad_list)
# print(f'After dropping invalid PVs\nFeature number is: {final_df.shape[1]}')




# Define column groups
vcc_profile = ['CAMR:IN20:186:XRMS', 'CAMR:IN20:186:YRMS']
RF_ampls = ['ACCL:LI21:1:L1S_S_AV', 'ACCL:LI21:180:L1X_S_AV', 'ACCL:LI22:1:ADES', 'ACCL:LI25:1:ADES']
RF_phases = ['ACCL:LI21:1:L1S_S_PV', 'ACCL:LI21:180:L1X_S_PV', 'ACCL:LI22:1:PDES', 'ACCL:LI25:1:PDES']
blen = ['BLEN:LI21:265:AIMAX1H', 'BLEN:LI24:886:BIMAX1H']
bcharge = ['SIOC:SYS0:ML00:CALC038', 'SIOC:SYS0:ML00:CALC252']
hxr_energy = ['BEND:DMPH:400:BACT','SIOC:SYS0:ML00:AO627']
sxr_energy = ['BEND:DMPS:400:BDES','SIOC:SYS0:ML00:AO628']
hxr_intensity = ['GDET:FEE1:241:ENRC1H']
laser_iris_status = ['IRIS:LR20:130:CONFG_SEL']

beam_status = ['XRMS on VCC', 'YRMS on VCC', 'Bunch length at BC1', 'Bunch length at BC2', 'Charge at gun [pC]', 'Charge after BC1 [pC]', 
               'HXR electron energy [GeV]', 'HXR photon energy [eV]', 'laser_iris_status']
status_from_archive = vcc_profile + blen + bcharge + hxr_energy + laser_iris_status
bpm_signal = ['BPMS:DMPH:381:TMIT1H']

# Load quads from lcls-live
quads = pd.read_csv('quad_mapping.csv')
quads_list = quads['device_name'].tolist()
quads_list = [quad + ':BCTRL' for quad in quads_list]
# quads_list.extend(['SOLN:IN20:111:BCTRL', 'SOLN:IN20:121:BCTRL', 'SOLN:IN20:311:BCTRL','QUAD:IN20:121:BCTRL',
                   # 'QUAD:IN20:122:BCTRL', 'QUAD:IN20:361:BCTRL','QUAD:IN20:371:BCTRL', 'QUAD:IN20:425:BCTRL', 
                   # 'QUAD:IN20:441:BCTRL', 'QUAD:IN20:511:BCTRL', 'QUAD:IN20:525:BCTRL'])
quads_list.extend(['SOLN:IN20:121:BCTRL', 'SOLN:IN20:311:BCTRL','QUAD:IN20:121:BCTRL',
                   'QUAD:IN20:122:BCTRL', 'QUAD:IN20:361:BCTRL','QUAD:IN20:371:BCTRL', 'QUAD:IN20:425:BCTRL', 
                   'QUAD:IN20:441:BCTRL', 'QUAD:IN20:511:BCTRL', 'QUAD:IN20:525:BCTRL'])

undh_corr_x = ['XCOR:UNDH:1380:BCTRL', 'XCOR:UNDH:1480:BCTRL', 'XCOR:UNDH:1580:BCTRL', 'XCOR:UNDH:1680:BCTRL',
                 'XCOR:UNDH:1780:BCTRL', 'XCOR:UNDH:1880:BCTRL', 'XCOR:UNDH:1980:BCTRL', 'XCOR:UNDH:2080:BCTRL',
                 'XCOR:UNDH:2180:BCTRL', 'XCOR:UNDH:2280:BCTRL', 'XCOR:UNDH:2380:BCTRL', 'XCOR:UNDH:2480:BCTRL',
                 'XCOR:UNDH:2580:BCTRL', 'XCOR:UNDH:2680:BCTRL', 'XCOR:UNDH:2780:BCTRL', 'XCOR:UNDH:2880:BCTRL',
                 'XCOR:UNDH:2980:BCTRL', 'XCOR:UNDH:3080:BCTRL', 'XCOR:UNDH:3180:BCTRL', 'XCOR:UNDH:3280:BCTRL',
                 'XCOR:UNDH:3380:BCTRL', 'XCOR:UNDH:3480:BCTRL', 'XCOR:UNDH:3580:BCTRL', 'XCOR:UNDH:3680:BCTRL',
                 'XCOR:UNDH:3780:BCTRL', 'XCOR:UNDH:3880:BCTRL', 'XCOR:UNDH:3980:BCTRL', 'XCOR:UNDH:4080:BCTRL',
                 'XCOR:UNDH:4180:BCTRL', 'XCOR:UNDH:4280:BCTRL', 'XCOR:UNDH:4380:BCTRL', 'XCOR:UNDH:4480:BCTRL',
                 'XCOR:UNDH:4580:BCTRL', 'XCOR:UNDH:4680:BCTRL', 'XCOR:UNDH:4780:BCTRL']
undh_corr_y = ['YCOR:UNDH:1380:BCTRL', 'YCOR:UNDH:1480:BCTRL', 'YCOR:UNDH:1580:BCTRL', 'YCOR:UNDH:1680:BCTRL',
                 'YCOR:UNDH:1780:BCTRL', 'YCOR:UNDH:1880:BCTRL', 'YCOR:UNDH:1980:BCTRL', 'YCOR:UNDH:2080:BCTRL',
                 'YCOR:UNDH:2180:BCTRL', 'YCOR:UNDH:2280:BCTRL', 'YCOR:UNDH:2380:BCTRL', 'YCOR:UNDH:2480:BCTRL',
                 'YCOR:UNDH:2580:BCTRL', 'YCOR:UNDH:2680:BCTRL', 'YCOR:UNDH:2780:BCTRL', 'YCOR:UNDH:2880:BCTRL',
                 'YCOR:UNDH:2980:BCTRL', 'YCOR:UNDH:3080:BCTRL', 'YCOR:UNDH:3180:BCTRL', 'YCOR:UNDH:3280:BCTRL',
                 'YCOR:UNDH:3380:BCTRL', 'YCOR:UNDH:3480:BCTRL', 'YCOR:UNDH:3580:BCTRL', 'YCOR:UNDH:3680:BCTRL',
                 'YCOR:UNDH:3780:BCTRL', 'YCOR:UNDH:3880:BCTRL', 'YCOR:UNDH:3980:BCTRL', 'YCOR:UNDH:4080:BCTRL',
                 'YCOR:UNDH:4180:BCTRL', 'YCOR:UNDH:4280:BCTRL', 'YCOR:UNDH:4380:BCTRL', 'YCOR:UNDH:4480:BCTRL',
                 'YCOR:UNDH:4580:BCTRL', 'YCOR:UNDH:4680:BCTRL', 'YCOR:UNDH:4780:BCTRL']

undh_shifter = ['PHAS:UNDH:1495:GapDes', 'PHAS:UNDH:1595:GapDes', 'PHAS:UNDH:1695:GapDes', 'PHAS:UNDH:1795:GapDes',
                 'PHAS:UNDH:1895:GapDes', 'PHAS:UNDH:1995:GapDes', 'PHAS:UNDH:2095:GapDes', 'PHAS:UNDH:2295:GapDes',
                 'PHAS:UNDH:2395:GapDes', 'PHAS:UNDH:2495:GapDes', 'PHAS:UNDH:2595:GapDes', 'PHAS:UNDH:2695:GapDes',
                 'PHAS:UNDH:2795:GapDes', 'PHAS:UNDH:2995:GapDes', 'PHAS:UNDH:3095:GapDes', 'PHAS:UNDH:3195:GapDes',
                 'PHAS:UNDH:3295:GapDes', 'PHAS:UNDH:3395:GapDes', 'PHAS:UNDH:3495:GapDes', 'PHAS:UNDH:3595:GapDes',
                 'PHAS:UNDH:3695:GapDes', 'PHAS:UNDH:3795:GapDes', 'PHAS:UNDH:3895:GapDes', 'PHAS:UNDH:3995:GapDes',
                 'PHAS:UNDH:4095:GapDes', 'PHAS:UNDH:4195:GapDes', 'PHAS:UNDH:4295:GapDes', 'PHAS:UNDH:4395:GapDes',
                 'PHAS:UNDH:4495:GapDes', 'PHAS:UNDH:4595:GapDes', 'PHAS:UNDH:4695:GapDes']

undh_gap = ['USEG:UNDH:1450:GapDes', 'USEG:UNDH:1550:GapDes', 'USEG:UNDH:1650:GapDes', 'USEG:UNDH:1750:GapDes',
                 'USEG:UNDH:1850:GapDes', 'USEG:UNDH:2050:GapDes', 'USEG:UNDH:2250:GapDes',
                 'USEG:UNDH:2350:GapDes', 'USEG:UNDH:2450:GapDes', 'USEG:UNDH:2550:GapDes', 'USEG:UNDH:2650:GapDes',
                 'USEG:UNDH:2750:GapDes', 'USEG:UNDH:3050:GapDes', 'USEG:UNDH:3150:GapDes',
                 'USEG:UNDH:3250:GapDes', 'USEG:UNDH:3350:GapDes', 'USEG:UNDH:3450:GapDes', 'USEG:UNDH:3550:GapDes',
                 'USEG:UNDH:3650:GapDes', 'USEG:UNDH:3750:GapDes', 'USEG:UNDH:3850:GapDes', 'USEG:UNDH:3500:GapDes',
                 'USEG:UNDH:4050:GapDes', 'USEG:UNDH:4150:GapDes', 'USEG:UNDH:4250:GapDes', 'USEG:UNDH:4350:GapDes',
                 'USEG:UNDH:4450:GapDes', 'USEG:UNDH:4550:GapDes', 'USEG:UNDH:4650:GapDes'] # 'USEG:UNDH:1950:GapDes'

quads_list = list(filter(lambda x: x not in invalid_quad_list, quads_list))
input_cols = quads_list + RF_ampls + RF_phases + ['XRMS on VCC', 'YRMS on VCC'] + undh_corr_x + undh_corr_y + undh_shifter + undh_gap #['HXR electron energy [GeV]', 'HXR photon energy [eV]']
output_cols = ['hxr_pulse_intensity']
input_size = len(input_cols)
output_size = len(output_cols)


final_df = final_df.drop(columns=invalid_quad_list)
print(f'After dropping invalid PVs\nFeature number is: {final_df.shape[1]}')

input_cols = [col for col in input_cols if col not in invalid_quad_list]
print(input_cols)
input_variables = []
output_variables = []
for col in input_cols:
    lower_bound, default_value, upper_bound = final_df[col].quantile([0, 0.5, 1])
    # variable specification
    input_variables.append(ScalarInputVariable(name=col, default=default_value, value_range=[lower_bound, upper_bound]))
for col in output_cols:
    output_variables.append(ScalarOutputVariable(name=col))
    
# Define Dataset class
class MyDataset(Dataset):
    def __init__(self, dataframe, input_cols, output_cols):
        self.features = dataframe[input_cols].values
        self.outputs = dataframe[output_cols].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

# Split dataset and apply scaling
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=39)

input_mins = final_df[input_cols].min()
input_maxs = final_df[input_cols].max()
output_mins = final_df[output_cols].min()
output_maxs = final_df[output_cols].max()

input_scaler = AffineInputTransform(d=input_size,coefficient=torch.tensor(input_maxs.values-input_mins.values, dtype=torch.float32),
                                                     offset=torch.tensor(input_mins.values, dtype=torch.float32))
output_scaler = AffineInputTransform(d=output_size,coefficient=torch.tensor(output_maxs.values-output_mins.values, dtype=torch.float32),
                                                     offset=torch.tensor(output_mins.values, dtype=torch.float32))

train_df[input_cols] = input_scaler.transform(torch.tensor(train_df[input_cols].values, dtype=torch.float32))
test_df[input_cols] = input_scaler.transform(torch.tensor(test_df[input_cols].values, dtype=torch.float32))
train_df[output_cols] = output_scaler.transform(torch.tensor(train_df[output_cols].values, dtype=torch.float32))
test_df[output_cols] = output_scaler.transform(torch.tensor(test_df[output_cols].values, dtype=torch.float32))


def initialize_data_loaders(batch_size):

    # Create DataLoader instances
    train_dataset = MyDataset(train_df, input_cols, output_cols)
    test_dataset = MyDataset(test_df, input_cols, output_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
# Define model creation function
def create_model(version: int = 0):
    if version == 5:
        model = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Dropout(p=0.05),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Dropout(p=0.05),
                nn.Linear(64, 16),
                nn.ELU(),
                nn.Dropout(p=0.05),
                nn.Linear(16, 16),
                nn.ELU(),
                nn.Linear(16, output_size)
            )
    else:
        raise ValueError(f"Unknown model version {version}.")
    return model.float()


        
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=80, device='cpu'):
    model.to(device)
    train_losses, test_losses = [], []
    best_loss = float('inf')
    best_model_state = None
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        t_avg = (time.time() - t0) / (epoch + 1)
        t_r = (num_epochs - epoch - 1) * t_avg / 60  # in minutes
        t_info = "{:.2f} sec".format(60 * t_r) if t_r <= 1.0 else "{:.2f} min".format(t_r)
        info = f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.2e}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, ETA: {t_info}"
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            info = "\033[0;32m" + info + '\x1b[0m'  # Green color for best model
        print(info)
        
        scheduler.step(test_loss)

    return train_losses, test_losses, best_model_state
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a FEL NN model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model.")
    args = parser.parse_args()

    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize data loaders
    train_loader, test_loader = initialize_data_loaders(args.batch_size)

    # Initialize the model
    model = create_model(5).to(device)

    # Load model weights if a saved model exists
    if args.model_path and os.path.exists(args.model_path):
        try:
            print(f"Model path: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
            print("Training from scratch.")
    else:
        print(f"No saved model found at {args.model_path}. Training from scratch.")

    # Initialize the optimizer and loss function
    # Set up model and training parameters
    lr = 1e-5
    weight_decay = 1e-6
    n_epochs = 50
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=4, verbose=True)

    # Start training
    print("Starting training...")
    train_loss_values, test_loss_values, best_model_state = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, device)

    # After training, create a directory for saving model artifacts
    model_path = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/model/'
    nn_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_nn/'
    nn_path = os.path.join(model_path, nn_folder_name)
    os.makedirs(nn_path, exist_ok=True)

    # Define paths for saving model artifacts
    best_model_path = os.path.join(nn_path, 'best_lcls_fel_model.pt')
    final_model_path = os.path.join(nn_path, 'final_lcls_fel_model.pt')
    input_scaler_path = os.path.join(nn_path, 'lcls_fel_input_scaler.pt')
    output_scaler_path = os.path.join(nn_path, 'lcls_fel_output_scaler.pt')
    config_path = os.path.join(nn_path, 'feature_config.yml')
    train_losses_path = os.path.join(nn_path, 'train_losses.npy')
    test_losses_path = os.path.join(nn_path, 'test_losses.npy')

    # Save the best model
    torch.save(best_model_state, best_model_path)

    # Save the final model
    torch.save(model, final_model_path)

    # Save scalers
    torch.save(input_scaler, input_scaler_path)
    torch.save(output_scaler, output_scaler_path)

    # Save feature configuration
    yaml_dump = variables_as_yaml(input_variables, output_variables, config_path)

    # Save losses
    np.save(train_losses_path, np.array(train_loss_values))
    np.save(test_losses_path, np.array(test_loss_values))

    print(f"Model and all artifacts saved in: {nn_path}")

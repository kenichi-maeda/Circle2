import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import ace_tools as tools

class CircleNet(torch.nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()

        self.input_layer = torch.nn.Linear(10, 1024)
        self.batchnorm1 = torch.nn.LayerNorm(1024)
        self.activation = torch.nn.SiLU()  # Swish activation

        self.hidden1 = torch.nn.Linear(1024, 1024)
        self.batchnorm2 = torch.nn.LayerNorm(1024)

        self.hidden2 = torch.nn.Linear(1024, 512)
        self.batchnorm3 = torch.nn.LayerNorm(512)

        self.hidden3 = torch.nn.Linear(512, 256)
        self.batchnorm4 = torch.nn.LayerNorm(256)

        self.hidden4 = torch.nn.Linear(256, 128)
        self.batchnorm5 = torch.nn.LayerNorm(128)

        self.hidden5 = torch.nn.Linear(128, 64)
        self.batchnorm6 = torch.nn.LayerNorm(64)

        self.skip_proj = torch.nn.Linear(1024, 64)
        self.output_layer = torch.nn.Linear(64, 12)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.batchnorm1(x)
        x = self.activation(x)

        res1 = self.skip_proj(x)  # Skip connection

        x = self.hidden1(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden3(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden4(x)
        x = self.batchnorm5(x)
        x = self.activation(x)

        x = self.hidden5(x)
        x = self.batchnorm6(x)
        x = self.activation(x)

        x = x + res1  # Skip connection

        x = self.output_layer(x)
        return x

"""
# Load the trained model architecture
class CircleNet(torch.nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(10, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 12)
        )

    def forward(self, x):
        return self.network(x)
"""
# Instantiate and load the trained model
model = CircleNet()
model.load_state_dict(torch.load('circle_model_100M_multi_4.pth'))
model.eval()

# Test cases
old_input_data = [
    [19.941704187674112,69.45842885049191,13.253232197383625,70.90101127143697,38.138703586090706,91.50734318603911,37.19476989969272,82.63540680049091,41.73908091938616,20.578101417633754],
    [15.019460058796753,35.582624562270084,99.51128781582703,66.9138073188445,5.801114531264162,86.86953602786424,36.767022182087636,43.39785777841172,61.91272468993523,54.2057774165661],
    [14.658421038998792,83.18054479530319,26.599065080320738,95.24359185324612,84.30287640974818,39.86435905113197,16.980730754094985,56.462059699653636,41.49911444864009,88.48704166451013],
    [95.27748142512532,98.45128075735828,57.7547349512671,1.0383301779532617,10.286384507042957,76.55060512008176,76.81703594307817,70.39180909135771,84.32378741867383,47.5178052173073],
    [96.55797593736067,45.2537358725256,25.838504301281507,26.18340682604938,22.222441174740503,21.210474940012013,48.02398786702462,70.35280822778189,5.596780536056611,80.66604287703063]
]

old_desired_outputs = [
    [20.364052103149145,87.64332604766571,18.18980109384062,18.598903337671214,89.100105756608,19.687523607783245,37.62250871534443,51.47169179759889,31.166650336943484,57.5694593753209,56.93765213130868,39.65624553728102],
    [30.542951763584927,64.84473956233724,33.12476670085063,55.31439823074246,89.3743081382571,49.57659877457298,-114.12088445848691,429.1095667032678,414.17470078806946,357.6772244940786,-758.8822707196389,865.2102711844559],
    [26.282804992319868,74.73852523327304,20.507505395166827,38.24880172157184,71.77079934174091,26.20473911459148,55.412565625402706,67.51395825322939,39.98937850545586,36.88802831673488,41.27583567819556,47.43585232618658],
    [54.57018972468396,80.56116144545223,44.46504207328557,40.73221899292046,43.013535267401714,45.2955173483362,35.9775368500548,44.3203937492337,48.45186667330585,226.66489309713336,-8.091393476401436,169.15671248505265],
    [21.342869515973234,53.01683355178984,31.818518105718248,59.46483160619621,33.00139026519318,39.06432297191538,52.716762764887896,67.1708654185534,49.014411553864086,-114.05530258120287,124.10589314179768,170.7603306694261]
]

input_data = [
    [53.437978803021224, 23.106973052094514, 22.5076116009435, 52.96921004149727, 57.13569106310729, 77.19434405432402, 24.117136535588067, 33.559650282263675, 19.632419439284643, 97.37654208910517],
    [16.44611770666764, 55.502801151525226, 88.1444844449124, 21.998265627169943, 61.498606819167456, 55.17506591347515, 12.134727289581592, 9.416223938956737, 67.97935145717344, 1.4992308249505593],
    [4.823382203895433, 37.00012487535884, 0.11323404315819463, 40.48174352707492, 82.46090519916592, 10.829413444038693, 98.53204780970907, 40.542207786488106, 24.697152916499267, 22.743208037668396],
    [11.008522847639025, 58.516611085247206, 5.320238053280047, 61.63510508941618, 49.71779580647359, 58.19179552213159, 70.38140799599705, 96.36523314355776, 75.06944119115072, 2.895641674616256],
    [11.48223909742211, 20.33856814266969, 95.73760786848601, 24.442167546708184, 10.47435788676908, 35.78633771550699, 45.28134119019366, 39.65787924754607, 51.6455971606942, 77.4163372628727]
]

desired_outputs = [
    [46.763031538107356, 50.733392964822265, 28.421365171007206, 53.34252767321936, 45.75466083413127, 31.66767704536917, 26.830577651918183, 65.8163638441717, 32.37063997710968, 77.3428014886355, 78.81630758155595, 60.62153496362578],
    [43.49767647490103, 29.727179564179444, 37.365351597505814, 38.75311344965848, 25.199651589817222, 37.62821989460909, 49.693076290915954, 18.404788276776515, 38.61895737190498, 51.7655168044402, 37.6168162258714, 39.59000390740639],
    [51.37244501916389, 46.847330540775104, 47.579225596859125, 49.296697498339526, 82.74058285496722, 64.84460331274897, 73.03123629006734, 111.0998790315998, 100.71288361007178, 87.34781287092697, 153.57115637044478, 142.82537253844646],
    [30.122195752108478, 93.47847831442266, 40.36257578551615, 30.722883691638106, 101.22366559951466, 47.03773515776473, 58.866513426535064, 48.935333958028735, 48.80766470174804, 23.87146779576945, 12.882455814093476, 52.16290766743402],
    [53.199502788458084, 30.817152130803606, 43.01314697001145, 53.656719334453136, 34.25361051253799, 43.20955441664992, 8.18902116516512, 65.32543886211015, 45.107247992722726, 39.826075677266395, -69.69807657499615, 109.49194032102946]
]


"""
# Convert input data to tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Model prediction
with torch.no_grad():
    predictions = model(input_tensor).numpy().reshape(-1, 4, 3)

# Create DataFrame for visualization
data = []
for i, (inp, pred, desired) in enumerate(zip(input_data, predictions, desired_outputs)):
    for j in range(4):
        data.append([
            i + 1,  # Test case index
            inp[j * 2], inp[j * 2 + 1],  # Input coordinates
            pred[j][0], pred[j][1], pred[j][2],  # Predicted output
            desired[j * 3], desired[j * 3 + 1], desired[j * 3 + 2]  # Desired output
        ])

columns = ["Test Case", "x_input", "y_input", "X_pred", "Y_pred", "R_pred", "X_desired", "Y_desired", "R_desired"]
df = pd.DataFrame(data, columns=columns)

# Display table
print(df.to_string())
"""
# Convert input data to tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Model prediction
with torch.no_grad():
    predictions = model(input_tensor).numpy()

# Create and display tables for each test case
for i, (pred, desired) in enumerate(zip(predictions, desired_outputs)):
    df = pd.DataFrame(
        [pred, desired],
        index=['Predicted', 'Ground Truth'],
        columns=[f'X{j//3+1}' if j % 3 == 0 else f'Y{j//3+1}' if j % 3 == 1 else f'R{j//3+1}' for j in range(12)]
    )
    print(f"\nTest Case {i+1}")
    print(df.to_string())
    print("=" * 50)
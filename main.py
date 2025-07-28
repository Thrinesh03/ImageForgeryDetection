

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageChops
import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



dataset_path = "/content/Dataset"
output_path = "/content"

path_original = 'Original/'
path_tampered = 'Forged/'

image_size = (224, 224)


def ELA(img_path, quality=100, threshold=60):
    TEMP = 'ela_temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()
    WIDTH, HEIGHT = diff.size

    for x in range(WIDTH):
        for y in range(HEIGHT):
            r, g, b = d[x, y]
            modified_intensity = int(0.2989 * r + 0.587 * g + 0.114 * b)
            d[x, y] = (modified_intensity * SCALE,) * 3

    calculated_threshold = threshold * (quality / 100)
    binary_mask = diff.point(lambda p: 255 if p >= calculated_threshold else 0)
    return binary_mask


def resize_and_save(input_list, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in tqdm(input_list):
        try:
            img = Image.open(os.path.join(input_folder, fname)).convert("RGB")
            img = img.resize(image_size, PIL.Image.ANTIALIAS)
            img.save(os.path.join(output_folder, fname))
        except:
            print("Error processing:", fname)

resized_fake_path = output_path + "resized/fake/"
resized_real_path = output_path + "resized/real/"

os.makedirs(resized_fake_path, exist_ok=True)
os.makedirs(resized_real_path, exist_ok=True)

total_original = os.listdir(os.path.join(dataset_path, path_original))
total_tampered = os.listdir(os.path.join(dataset_path, path_tampered))

resize_and_save(total_tampered, dataset_path + path_tampered, resized_fake_path)
resize_and_save(total_original, dataset_path + path_original, resized_real_path)


ela_fake_path = output_path + "ela/fake/"
ela_real_path = output_path + "ela/real/"

os.makedirs(ela_fake_path, exist_ok=True)
os.makedirs(ela_real_path, exist_ok=True)

for fname in tqdm(os.listdir(resized_fake_path)):
    ela_img = ELA(os.path.join(resized_fake_path, fname))
    ela_img.save(os.path.join(ela_fake_path, fname))

for fname in tqdm(os.listdir(resized_real_path)):
    ela_img = ELA(os.path.join(resized_real_path, fname))
    ela_img.save(os.path.join(ela_real_path, fname))


X = []
Y = []

for fname in tqdm(os.listdir(ela_real_path)):
    img = Image.open(os.path.join(ela_real_path, fname)).resize(image_size)
    X.append(np.array(img).flatten())  # Flatten the image for ML model
    Y.append(0)

for fname in tqdm(os.listdir(ela_fake_path)):
    img = Image.open(os.path.join(ela_fake_path, fname)).resize(image_size)
    X.append(np.array(img).flatten())
    Y.append(1)

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# Support Vector Machine

svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

from PIL import Image
import os

# Chemin du dataset
dataset_path = "dataset"
classes = ["normal", "defaut"]

for cls in classes:
    images = os.listdir(f"{dataset_path}/{cls}")
    print(f"{cls} contient {len(images)} images")
    # Afficher la première image
    img = Image.open(f"{dataset_path}/{cls}/{images[0]}")
    img.show()
    break
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=10,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=10,
    class_mode='binary',
    subset='validation'
)

# Créer un modèle simple CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_gen, validation_data=val_gen, epochs=2)

# Tester sur une image de validation
img = val_gen[0][0][0]
pred = model.predict(np.expand_dims(img, axis=0))
print("Prediction :", pred)
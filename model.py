import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras import losses # type: ignore
from tensorflow.keras.saving import register_keras_serializable# type: ignore


class FaceDetector:
    def __init__(self, input_shape=(224, 224, 3), max_faces=5):
        self.input_shape = input_shape
        self.max_faces = max_faces
        self.model = self.build_model()
        
    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Feature extraction backbone
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Detection head for multiple faces
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output: 4 coordinates per face * max_faces + confidence scores
        bbox_output = layers.Dense(self.max_faces * 4, activation='sigmoid', name='bbox')(x)
        confidence_output = layers.Dense(self.max_faces, activation='sigmoid', name='confidence')(x)
        
        model = models.Model(inputs=inputs, outputs=[bbox_output, confidence_output])
        
        model.compile(optimizer='adam',
                      loss={'bbox': 'mse', 'confidence': 'binary_crossentropy'},
                      metrics={'bbox': 'accuracy'})
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        return history
    
    def save(self, path):
        self.model.save(path)
    
    @staticmethod
    def load(path):
        detector = FaceDetector()
        detector.model = tf.keras.models.load_model(path)
        return detector
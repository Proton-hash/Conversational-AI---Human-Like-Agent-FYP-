import tensorflow as tf
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import argparse


def load_image_dataset(train_dir, test_dir, batch_size=16):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    df_train = datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical'
    )
    df_test = datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical'
    )
    return df_train, df_test

def load_text_dataset(file_path):
    text_df = pd.read_csv(file_path)
    texts = text_df['text'].values
    emotions = text_df['Emotion'].values
    return texts, emotions, text_df

def preprocess_text_data(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    max_seq_length = max([len(seq) for seq in sequences])
    data = pad_sequences(sequences, maxlen=max_seq_length)

    w2v_model = Word2Vec(sentences=[text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = w2v_model.wv

    embedding_dim = 100
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)
    
    return data, embedding_matrix, word_index, max_seq_length, tokenizer


def define_image_model():
    image_input = Input(shape=(48, 48, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_features = Dense(128, activation='relu')(x)
    return image_input, image_features

def define_text_model(word_index, embedding_matrix, max_seq_length):
    text_input = Input(shape=(max_seq_length,))
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(word_index) + 1,
                                                output_dim=100,
                                                weights=[embedding_matrix],
                                                input_length=max_seq_length,
                                                trainable=False)(text_input)
    y = Conv1D(32, 5, activation='relu')(embedding_layer)
    y = MaxPooling1D(5)(y)
    y = Flatten()(y)
    text_features = Dense(128, activation='relu')(y)
    return text_input, text_features

def define_combined_model(image_input, image_features, text_input, text_features):
    combined_features = concatenate([image_features, text_features])
    z = Dense(128, activation='relu')(combined_features)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.5)(z)
    output = Dense(7, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image_dataset(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch)
        labels.append(label_batch)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

def train_model(model, train_images, text_train_data, train_labels, val_images, val_text_data, val_labels):
    min_samples = min(train_images.shape[0], text_train_data.shape[0])
    train_images = train_images[:min_samples]
    text_train_data = text_train_data[:min_samples]
    train_labels = train_labels[:min_samples]

    model.fit(
        [train_images, text_train_data],
        to_categorical(train_labels),
        validation_data=([val_images, val_text_data], to_categorical(val_labels)),
        epochs=20,
        batch_size=32
    )

def predict_emotion(model, tokenizer, text_df, text, image_path, max_seq_length):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_seq_length)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(48, 48))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    predictions = model.predict([image, padded_seq])
    return predictions[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Predict with the Emotion Classification Model.')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--text', type=str, help='Text input for prediction')
    parser.add_argument('--image', type=str, help='Image file path for prediction')
    args = parser.parse_args()

    train_dir = "G:/MyProject/train"
    test_dir = "G:/MyProject/test"
    text_file_path = "balanced_filtered_dataset.csv"
    model_file_path = "emotion_classification_model.h5"

    if args.mode == 'train':
        df_train, df_test = load_image_dataset(train_dir, test_dir, batch_size=16)

        texts, emotions, text_df = load_text_dataset(text_file_path)
        text_data, embedding_matrix, word_index, max_seq_length, tokenizer = preprocess_text_data(texts)
        
        image_input, image_features = define_image_model()
        text_input, text_features = define_text_model(word_index, embedding_matrix, max_seq_length)
        model = define_combined_model(image_input, image_features, text_input, text_features)

        train_images, train_labels = preprocess_image_dataset(df_train)
        test_images, test_labels = preprocess_image_dataset(df_test)

        text_train_data, text_test_data, text_train_labels, text_test_labels = train_test_split(text_data, pd.get_dummies(emotions).values, test_size=0.2, random_state=42)

        train_model(model, train_images, text_train_data, train_labels, test_images, text_test_data, test_labels)

        model.save(model_file_path)
        print(f"Model saved to {model_file_path}")

    elif args.mode == 'predict':
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found. Please train the model first.")

        model = tf.keras.models.load_model(model_file_path)
        print(f"Model loaded from {model_file_path}")

        texts, emotions, text_df = load_text_dataset(text_file_path)
        _, _, _, max_seq_length, tokenizer = preprocess_text_data(texts)

        if args.text is None or args.image is None:
            raise ValueError("For prediction, both --text and --image arguments must be provided.")

        predictions = predict_emotion(model, tokenizer, text_df, args.text, args.image, max_seq_length)

        emotion_labels = text_df['Emotion'].unique()
        for emotion, confidence in zip(emotion_labels, predictions):
            print(f"{emotion}: {confidence:.2f}%")

from keras import Input
from keras.layers import LSTM, Dropout, BatchNormalization, concatenate, Dense, Reshape
from keras.models import Model
from keras.optimizers import Adam


def create_model(input_length, lstm_units=256, dense_units=64):
    # Define inputs
    input_1 = Input(shape=(input_length,))
    input_2 = Input(shape=(input_length,))

    # Reshape inputs for LSTM layer
    reshaped_input_1 = Reshape((input_length, 1))(input_1)
    reshaped_input_2 = Reshape((input_length, 1))(input_2)

    # LSTM layers
    lstm_layer1 = LSTM(units=lstm_units, activation='relu', return_sequences=True)

    # Branch 1
    x1 = lstm_layer1(reshaped_input_1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)

    # Branch 2
    x2 = lstm_layer1(reshaped_input_2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.4)(x2)

    # Merge the two branches
    merged = concatenate([x1, x2], axis=-1)

    merged = Dense(units=dense_units, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(units=dense_units, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(units=1, activation='sigmoid')(merged)

    # Create the model
    model = Model(inputs=[input_1, input_2], outputs=merged)
    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

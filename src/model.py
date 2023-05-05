import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn import tree


def create_model(vocab_size, embedding_dim, input_length):
    # Input layer
    input_1 = Input(shape=(input_length,))
    input_2 = Input(shape=(input_length,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)

    # LSTM layer
    lstm_layer = LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)

    # Branch 1
    x1 = embedding_layer(input_1)
    x1 = lstm_layer(x1)
    x1 = Dropout(0.2)(x1)

    # Branch 2
    x2 = embedding_layer(input_2)
    x2 = lstm_layer(x2)
    x2 = Dropout(0.2)(x2)

    # Merge the two branches
    merged = concatenate([x1, x2])
    merged = Dense(units=64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(units=1, activation='sigmoid')(merged)

    # Create the model
    model = Model(inputs=[input_1, input_2], outputs=merged)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''

    # Get a list of class names
    numclasses = len(model.classes_)
    if len(model.classes_) == 2:
        classes = [model.classes_[1]]  # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_

    # Create a plot for each class
    for i,c in enumerate(classes):
        # Plot the coefficients as bars
        fig = plt.figure(figsize=(8,len(column_names)/3))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16)
        rects = plt.barh(column_names, model.coef_[i],color="lightblue")

        # Annotate the bars with the coefficient values
        for rect in rects:
            width = round(rect.get_width(),4)
            plt.gca().annotate('  {}  '.format(width),
                               xy=(0, rect.get_y()),
                               xytext=(0,2),
                               textcoords="offset points",
                               ha='left' if width<0 else 'right', va='bottom')
        plt.show()
        #for pair in zip(X.columns, model_lr.coef_[i]):
        #    print (pair)

def decisionTreeSummary(model, column_names):
    '''Show a summary of the trained decision tree model'''

    # Plot the feature importances as bars
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Decision tree feature importance', fontsize=16)
    rects = plt.barh(column_names, model.feature_importances_,color="khaki")

    # Annotate the bars with the feature importance values
    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                           xy=(width, rect.get_y()),
                           xytext=(0,2),
                           textcoords="offset points",
                           ha='left', va='bottom')

    plt.show()

def linearRegressionSummary(model, column_names):
    '''Show a summary of the trained linear regression model'''

    # Plot the coeffients as bars
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Linear Regression Coefficients', fontsize=16)
    rects = plt.barh(column_names, model.coef_,color="lightblue")

    # Annotate the bars with the coefficient values
    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                           xy=(0, rect.get_y()),
                           xytext=(0,2),
                           textcoords="offset points",
                           ha='left' if width<0 else 'right', va='bottom')
    plt.show()


def viewDecisionTree(model, column_names):
    '''Visualise the decision tree'''

    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=column_names,
                                    class_names=model.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)
    # graph = graphviz.Source(dot_data)
    # return graph


def find_outliers(feature):
    '''Return a list of outliers in the data'''

    # Temporarily replace nulls with mean so they don't cause an error
    feature = feature.fillna(feature.mean())

    # Compute the quartiles
    quartile_1, quartile_3 = np.percentile(feature, [25, 75])

    # Compute the inter-quartile range
    iqr = quartile_3 - quartile_1

    # Compute the outlier boundaries
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    # Return rows where the feature is outside the outlier boundaries
    return np.where((feature > upper_bound) | (feature < lower_bound))
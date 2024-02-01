from matplotlib import pyplot as plt
import librosa
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_spectrogram(signal, name):
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for {name}")
    plt.xlabel("Time")
    plt.show()


def print_multiclass_confusion_matrix(confusion_matrix, axes, class_label, class_names, font_size=14):
    df_cm = pd.DataFrame(confusion_matrix,
                         index=class_names,
                         columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=font_size)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=font_size)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)


def plot_confusion_matrix(y_test, y_true, y_pred, image_path=f'Doc/figures/confusion_matrix.jpg'):
    labels = np.unique(y_test)
    labels_translated = np.array(['Гнев',
                                  'Спокойствие',
                                  'Отвращение',
                                  'Страх',
                                  'Счастье',
                                  'Нейтральность',
                                  'Грусть',
                                  'Удивление'])
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)

    df_cm_4 = pd.DataFrame(cm, columns=labels_translated, index=labels_translated)
    fig, ax = plt.subplots(dpi=110)
    sns.heatmap(df_cm_4, annot=True, fmt='.2f', linewidth=1.5)
    plt.show()
    print('Saving confusion matrix with name: ', image_path)
    fig.savefig(image_path, format='jpg', dpi=200, bbox_inches='tight', pad_inches=0.2)


def plot_confusion_matrix_eng(y_true, y_pred, image_path=f'Doc/figures/confusion_matrix_eng.jpg'):
    labels = np.unique(y_true)
    labels_translated = np.array(['Anger',
                                  'Calmness',
                                  'Disgust',
                                  'Fear',
                                  'Happiness',
                                  'Neutrality',
                                  'Sadness',
                                  'Surprise'])
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)

    df_cm_4 = pd.DataFrame(cm, columns=labels_translated, index=labels_translated)
    fig, ax = plt.subplots(dpi=110)
    sns.heatmap(df_cm_4, annot=True, fmt='.2f', linewidth=1.5)
    plt.show()
    print('Saving confusion matrix with name: ', image_path)
    fig.savefig(image_path, format='jpg', dpi=200, bbox_inches='tight', pad_inches=0.2)
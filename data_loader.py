import glob
import os
import platform
import time

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm


def get_emotions_dictionary():
    return {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprised'
    }


def get_actors():
    actors = {
        '01': '1',
        '02': '2',
        '03': '3',
        '04': '4',
        '05': '5',
        '06': '6',
        '07': '7',
        '08': '8',
        '09': '9',
        '10': '10',
        '11': '11',
        '12': '12',
        '13': '13',
        '14': '14',
        '15': '15',
        '16': '16',
        '17': '17',
        '18': '18',
        '19': '19',
        '20': '20',
        '21': '21',
        '22': '22',
        '23': '23',
        '24': '24'
    }
    return actors


def get_tess_emotions():
    # defined tess emotions to test on TESS dataset only
    return ['angry', 'disgust', 'fear', 'ps', 'happy', 'sad']


def get_ravdess_emotions():
    # defined RAVDESS emotions to test on RAVDESS dataset only
    return ['neutral', 'calm', 'angry', 'happy', 'disgust', 'sad', 'fear', 'surprised']


def get_observed_emotions():
    return ['sad', 'angry', 'happy', 'disgust', 'surprised', 'neutral', 'calm', 'fear']


# Function to perform voice activity detection
def detect_voice_activity(audio_file_path, silence_threshold=40):
    audio = AudioSegment.from_wav(audio_file_path)
    voice_segments = []
    current_segment = []

    for sample in audio:
        if abs(sample.dBFS) > silence_threshold:
            current_segment.append(sample)
        elif len(current_segment) > 0:
            voice_segments.append(AudioSegment(current_segment))
            current_segment = []

    return voice_segments


def extract_mffcs_with_vad(file_name, n_fft=8192):
    """
    Performs voice activity detection on a given sample.
    After that performs the calculation of supra-segment features
    based on MFCC to obtain a characteristic vector.

    Output:
    The resulting mean MFCC and standard deviation of MFCC
    provides a measure of the average squared deviation of
    each MFCC feature across timeline.
    """
    # Process voice segments and extract MFCCs
    X, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=n_fft).T        
    # VAD
    mfccs = mfccs[:,1:] # exclude coeff 0
    mel_frame_power = np.sum(mfccs**2,axis=1)/mfccs.shape[1]
    mel_uttr_power = np.sum(mel_frame_power)/mfccs.shape[0]
    r1,r2 = 5, 0.5
    vad_threshold = r1 + r2*np.log10(mel_uttr_power)
    vad_out = mel_frame_power>vad_threshold

    #voice_segments = detect_voice_activity(file_name)
    
    mfcc_features = mfccs[vad_out,:]
    # for ind in range(vad_out):
    #     if vad_out:
    #         mfcc_features.append(mfccs[ind,:])
        
    mean_mfcc = np.mean(mfcc_features, axis=0)
    sd_mfcc = np.std(mfcc_features, axis=0)
    if np.sum(np.isnan(sd_mfcc))>0:
        print('NaN!')

    return mean_mfcc, sd_mfcc


def extract_feature(file_name, mfcc_flag, n_fft=8192):
    """
    Performing the calculation of supra-segment features
    based on MFCC to obtain a characteristic vector.

    Output:
    The resulting mean MFCC and standard deviation of MFCC
    provides a measure of the average squared deviation of
    each MFCC feature across timeline.
    """
    X, sample_rate = librosa.load(file_name)
    if mfcc_flag:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40,
                                     n_fft=n_fft).T  # default n_fft = 2048 (window size)
        mean_mfcc = np.mean(mfccs, axis=0)
        sd_mfcc = np.std(mfccs, axis=0)
        return mean_mfcc, sd_mfcc
    else:
        return None

def extract_mfccs_with_delta(file_name, n_fft=8192):
    """
    Performing the calculation of supra-segment features
    based on MFCC to obtain a feature vector.

    Output:
    mean MFCC / SD of MFCC / mean delta MFCC / SD of delta MFCC    
    each MFCC feature averaged across time axis.
    """
    X, sample_rate = librosa.load(file_name)
    
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40,
                                     n_fft=n_fft).T  # default n_fft = 2048 (window size)
    mfcc_delta = librosa.feature.delta(mfccs)

    mean_mfcc = np.mean(mfccs, axis=0)
    sd_mfcc = np.std(mfccs, axis=0)

    mean_delta_mfcc = np.mean(mfcc_delta, axis=0)
    sd_delta_mfcc = np.std(mfcc_delta, axis=0)

    return mean_mfcc, sd_mfcc, mean_delta_mfcc, sd_delta_mfcc
    

def dataset_options():
    # choose datasets
    ravdess = True
    tess = False
    ravdess_speech = False
    ravdess_song = False
    n_fft = 8192  # 2048 -- default (others: 32768 / 8192 / 4096)
    data = {'ravdess': ravdess, 'ravdess_speech': ravdess_speech, 'ravdess_song': ravdess_song, 'tess': tess}
    print(data)
    return data, n_fft


def build_dataset(use_vad=False, use_delta_mfcc=False):
    X, y, ID = [], [], []

    # feature to extract
    mfcc = True

    data, n_fft = dataset_options()
    paths = []
    if data['ravdess']:
        if platform.system() == "Windows":
            paths.append("data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/**/*.wav")
        else:
            paths.append("data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/**/*.wav")
    elif data['ravdess_speech']:
        paths.append("./data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_*/*.wav")
    elif data['ravdess_song']:
        paths.append("./data/ravdess-emotional-song-audio/audio_song_actors_01-24/Actor_*/*.wav")

    for path in paths:
        for file in tqdm(glob.glob(path)):
            file_name = os.path.basename(file)
            splited_file_name = file_name.split("-")
            emotion = get_emotions_dictionary()[
                splited_file_name[2]]  # to get emotion according to filename. dictionary emotions is defined above.
            if emotion not in get_observed_emotions():  # options observed_emotions - RAVDESS and TESS, ravdess_emotions for RAVDESS only
                continue
            actor = np.array(get_actors()[splited_file_name[6].split(".")[0]])
            if use_vad:
                if use_delta_mfcc:
                    print('Error! No function mfcc with VAD')
                else:
                    mean_mfcc, sd_mfcc = extract_mffcs_with_vad(file, n_fft=n_fft)
            else:
                if use_delta_mfcc:
                    mean_mfcc, sd_mfcc, mean_mfcc_delta, sd_mfcc_delta = extract_mfccs_with_delta(file, n_fft=n_fft)
                    feature = np.hstack((mean_mfcc, sd_mfcc, mean_mfcc_delta, sd_mfcc_delta))
                else:
                    mean_mfcc, sd_mfcc = extract_feature(file, mfcc, n_fft=n_fft)
                    feature = np.hstack((mean_mfcc, sd_mfcc))
            X.append(feature)
            y.append(emotion)
            ID.append(actor)
    if data['tess']:
        for file in glob.glob(
                "./data/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/*AF_*/*.wav"):
            file_name = os.path.basename(file)
            emotion = file_name.split("_")[2][:-4]  # split and remove .wav
            if emotion == 'ps':
                emotion = 'surprised'
            if emotion not in get_observed_emotions():  # options observed_emotions - RAVDESS and TESS, ravdess_emotions for RAVDESS only
                continue
            feature = extract_feature(file, mfcc)
            X.append(feature)
            y.append(emotion)
    return {"X": X, "y": y, "ID": ID}


def generate_csv_dataset(use_vad=False, use_delta_mfcc=False):
    """
    Generates a data set containing combined features matrix (MFCCs and standard deviation of MFCC),
    class labels and actor IDs

    :return: X - features, y = class_labels, IDs = actor id
    """
    start_time = time.time()
    _, n_fft = dataset_options()
    print(f'INFO: n_fft={n_fft}')

    dataset = build_dataset(use_vad, use_delta_mfcc)

    print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
    X = pd.DataFrame(dataset["X"])
    y = pd.DataFrame(dataset["y"])
    ID = pd.DataFrame(dataset["ID"])

    # Standardize data
    X = (X - X.mean()) / X.std()

    print("X.shape = ", X.shape)
    print("y.shape = ", y.shape)
    print("ID.shape = ", ID.shape)

    if use_vad:
        X_path = f'data/feature_vector_based_mean_mfcc_and_std_mfcc_nfft_{n_fft}_vad.csv'
    else:
        if use_delta_mfcc:
            X_path = f'data/feature_mfcc_delta_nfft_{n_fft}.csv'
        else:
            X_path = f'data/feature_vector_based_mean_mfcc_and_std_mfcc_nfft_{n_fft}.csv'
    
    y_path = f'data/y_labels.csv'
    ID_path = f'data/IDs.csv'                
    X.to_csv(X_path)
    y.to_csv(y_path)
    ID.to_csv(ID_path)

    return X_path, y_path, ID_path


def load_dataset(should_generate_dataset=False,
                 use_vad=False,
                 X_path='data/feature_vector_based_mean_mfcc_and_std_mfcc.csv',
                 y_path='data/y_labels.csv',
                 ID_path='data/IDs.csv'):
    if should_generate_dataset:
        X_path, y_path, ID_path = generate_csv_dataset(use_vad)
    starting_time = time.time()
    # data = pd.read_csv('./data/characteristic_vector_using_mfcc_and_mean_square_deviation_20230722.csv')
    X = pd.read_csv(X_path)
    X = X.drop('Unnamed: 0', axis=1)
    y = pd.read_csv(y_path)
    y = y.drop('Unnamed: 0', axis=1)
    ID = pd.read_csv(ID_path)
    ID = ID.drop('Unnamed: 0', axis=1)

    print("data loaded in " + str(time.time() - starting_time) + "ms")
    print(X.head())
    print("X.shape = ", X.shape)
    print("X.columns = ", X.columns)

    return X, y, ID


def get_k_fold_group_member():
    return {
        '0': {2, 5, 14, 15, 16},
        '1': {3, 6, 7, 13, 18},
        '2': {10, 11, 12, 19, 20},
        '3': {8, 17, 21, 23, 24},
        '4': {1, 4, 9, 22}
    }


def get_custom_k_folds(X, y, ID, group_members):
    X_k_fold = dict()
    y_k_fold = dict()
    for k, members in group_members.items():
        fold_X = pd.DataFrame()
        fold_y = pd.DataFrame()

        for actor_ID in members:
            inds = ID[ID['0'] == actor_ID].index.tolist()
            fold_X = pd.concat([fold_X, X.loc[inds, :]])
            fold_y = pd.concat([fold_y, y.loc[inds, :]])
        X_k_fold[k] = fold_X
        y_k_fold[k] = fold_y
    return X_k_fold, y_k_fold

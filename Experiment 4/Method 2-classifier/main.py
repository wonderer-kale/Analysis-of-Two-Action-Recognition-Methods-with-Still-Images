import os
import time
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from extract_feature import extract_feature

actions = []
skiped_actions = ['look', 'hold', 'jump']  # jump is too ambiguous

def split_dataset():
    print('\n ======== Splitting dataset ======== ')
    root_dir = "data/all"
    datas = []
    labels = []
    ride_set = set()

    # Dataset Improvement
    # Load all image id, skip the actions that have less than 400 images
    for i, action in enumerate(os.listdir(root_dir)):
        # Skip the actions that are in the skiped_actions list
        if action in skiped_actions:
            continue

        action_path = os.path.join(root_dir, action)
        image_ids = os.listdir(action_path)

        # If there are less than 400 images, ignore this image
        # Ignore list: drink, kick, point, read
        if len(image_ids) < 400:
            print(f'{action} has less than 400 images, ignore')
            continue
        
        # Delete the images that are exist in both ride and sit from sit images
        if action == 'ride':
            ride_set = set(image_ids)
        if action == 'sit':
            image_ids = list(set(image_ids) - ride_set)
        
        # If there are enough images, add the first 400 images to the dataset
        datas += [os.path.join(action_path, image_id) for image_id in image_ids[:400]]
        labels += ([i] * 400)
        actions.append(action)
        print(f'{i} - {action}: {len(image_ids)} images')
    train_id_path, test_id_path, train_label, test_label = train_test_split(datas, labels, test_size=0.1, random_state=42, stratify=labels)
    train_id_path, train_label = zip(*sorted(zip(train_id_path, train_label)))
    test_id_path, test_label = zip(*sorted(zip(test_id_path, test_label)))
    print('Train dataset:', len(train_id_path))
    print('Test dataset:', len(test_id_path))

    return train_id_path, test_id_path, train_label, test_label


def extract_features(type, id_paths, labels):
    print(f'Extracting features on {type}...')
    X = [] # features
    y = [] # actions
    image_types = ['agent.jpg', 'object.jpg', 'intersection.jpg', 'union.jpg']
    action, prev_action = '', ''
    finish_extracting = True
    
    for id_path, label in zip(id_paths, labels):
        action = id_path.split('/')[2]

        # Skip the actions that are in the skiped_actions list
        if action in skiped_actions:
            continue
        # Skip the action if the features are already extracted
        if os.path.exists(f'features/all/{type}/{action}_X.npy'):
            continue

        finish_extracting = False
        # If the action is different from the previous one, save the features
        if action != prev_action:
            print(f'{prev_action} features saved!')
            np.save(f'features/all/{type}/{prev_action}_X.npy', X)
            np.save(f'features/all/{type}/{prev_action}_y.npy', y)
            X.clear()
            y.clear()
            print(f'\nProcessing {action}')
        
        real_image_type = os.listdir(id_path)
        images = []

        for image_type in image_types:
            # If there is no intersection, use white image
            if image_type not in real_image_type:
                image_path = 'white.jpeg'
            else:
                image_path = os.path.join(id_path, image_type)
            image = Image.open(image_path)
            images.append(image)
        components = extract_feature(images)
        X.append(components)
        y.append(label)

        prev_action = action

    # Save the last action and remove the first file
    if not finish_extracting: # means there is at least one action that is not saved
        np.save(f'features/all/{type}/{action}_X.npy', X)
        np.save(f'features/all/{type}/{action}_y.npy', y)
    
    # Remove the first file
    if os.path.exists(f'features/all/{type}/_X.npy'):
        os.remove(f'features/all/{type}/_X.npy')
    if os.path.exists(f'features/all/{type}/_y.npy'):
        os.remove(f'features/all/{type}/_Y.npy')

def train(train_id_path, train_label):
    print('\n ======== Training ======== ')
    
    # Load images and extract features using AlexNet
    extract_features('train', train_id_path, train_label)

    # Load features
    print('\nLoading features...')
    X = []
    X_catch_throw = []
    X_sit_ride = []
    X_snow_skate = []
    y = []
    y_catch_throw = []
    y_sit_ride = []
    y_snow_skate = []

    # Load features and labels. Specially process for the sub-classifiers
    for action in actions:
        if action in ['catch', 'throw']: # Only use agent to classify
            X_catch_throw.append(np.load(f'features/all/train/{action}_X.npy')[:, 0:4096])
            y_catch_throw.append(np.load(f'features/all/train/{action}_y.npy'))
        elif action in ['sit', 'ride']: # Only use object to classify
            X_sit_ride.append(np.load(f'features/all/train/{action}_X.npy')[:, 4096:8192])
            y_sit_ride.append(np.load(f'features/all/train/{action}_y.npy'))
        elif action in ['snowboard', 'skateboard']: # Only use object to classify
            X_snow_skate.append(np.load(f'features/all/train/{action}_X.npy')[:, 4096:8192])
            y_snow_skate.append(np.load(f'features/all/train/{action}_y.npy'))
        X.append(np.load(f'features/all/train/{action}_X.npy'))
        y.append(np.load(f'features/all/train/{action}_y.npy'))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    X_catch_throw = np.concatenate(X_catch_throw, axis=0)
    y_catch_throw = np.concatenate(y_catch_throw, axis=0)
    X_sit_ride = np.concatenate(X_sit_ride, axis=0)
    y_sit_ride = np.concatenate(y_sit_ride, axis=0)
    X_snow_skate = np.concatenate(X_snow_skate, axis=0)
    y_snow_skate = np.concatenate(y_snow_skate, axis=0)
    print('Total number of images', X.shape[0])
    
    # Train the classifiers
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                    tol=1e-2, max_iter=200000, class_weight='balanced',   # hyparameters (?)
                                    n_jobs=-1)
    classifier_catch_throw = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                                tol=1e-2, max_iter=50000, class_weight='balanced',   # hyparameters (?)
                                                n_jobs=-1)
    classifier_sit_ride = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                             tol=1e-2, max_iter=50000, class_weight='balanced',   # hyparameters (?)
                                             n_jobs=-1)
    classifier_snow_skate = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                               tol=1e-2, max_iter=50000, class_weight='balanced',   # hyparameters (?)
                                               n_jobs=-1)

    timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
    print('\nTraining the classifiers...')

    classifier.fit(X, y)
    with open(f'model/model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print('Classifier saved')

    classifier_catch_throw.fit(X_catch_throw, y_catch_throw)
    with open(f'model/model_catch_throw_{timestamp}.pkl', 'wb') as f:
        pickle.dump(classifier_catch_throw, f)
    print('Classifier_catch_throw saved')

    classifier_sit_ride.fit(X_sit_ride, y_sit_ride)
    with open(f'model/model_sit_ride_{timestamp}.pkl', 'wb') as f:
        pickle.dump(classifier_sit_ride, f)
    print('Classifier_sit_ride saved')

    classifier_snow_skate.fit(X_snow_skate, y_snow_skate)
    with open(f'model/model_snow_skate_{timestamp}.pkl', 'wb') as f:
        pickle.dump(classifier_snow_skate, f)
    print('Classifier_snow_skate saved')
    
    # Test the classifier in training dataset
    print('\nFinished training. The accuracy of the model on training dataset is: ')
    overall_score = 0
    for action in actions:
        X = np.load(f'features/all/train/{action}_X.npy')
        y = np.load(f'features/all/train/{action}_y.npy')

        score = 0.0

        # Predict the action using sub-classifiers if needed
        for X_i, y_i in zip(X, y):
            prob = classifier.predict_proba(X_i.reshape(1, -1))
            if prob[0][1] + prob[0][10] > 0.5: # catch or throw
                pred = classifier_catch_throw.predict(X_i[0:4096].reshape(1, -1))[0]
            elif prob[0][3] + prob[0][4] > 0.5: # sit or ride
                pred = classifier_sit_ride.predict(X_i[4096:8192].reshape(1, -1))[0]
            elif prob[0][5] + prob[0][7] > 0.5: # snowboard, or skateboard
                pred = classifier_snow_skate.predict(X_i[4096:8192].reshape(1, -1))[0]
            else:
                pred = classifier.predict(X_i.reshape(1, -1))[0]
            score += (1 if pred == y_i else 0)
        
        score /= len(X)
        overall_score += score
        print(f'The accuracy of action {action}: {score}')
    print(f'↳ The overall accuracy: {overall_score / (len(actions) - len(skiped_actions))}')


def test(timestamp, test_id_path, test_label):
    print('\n ======== Testing ======== ')

    # Load images and extract features using AlexNet
    extract_features('test', test_id_path, test_label)

    # Load the model
    print('\nLoading the model...')
    with open(f'model/model_{timestamp}.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open(f'model/model_catch_throw_{timestamp}.pkl', 'rb') as f:
        classifier_catch_throw = pickle.load(f)
    with open(f'model/model_sit_ride_{timestamp}.pkl', 'rb') as f:
        classifier_sit_ride = pickle.load(f)
    with open(f'model/model_snow_skate_{timestamp}.pkl', 'rb') as f:
        classifier_snow_skate = pickle.load(f)

    # Test the classifier in testing dataset
    print('\nLoading features...')
    y_pred = []
    y_true = []
    overall_score = 0
    for action in actions:
        X = np.load(f'features/all/test/{action}_X.npy')
        y = np.load(f'features/all/test/{action}_y.npy')
        
        score = 0.0
        prediction = []

        # Predict the action using sub-classifiers if needed
        for X_i, y_i in zip(X, y):
            prob = classifier.predict_proba(X_i.reshape(1, -1))
            if prob[0][1] + prob[0][10] > 0.5: # catch or throw
                pred = classifier_catch_throw.predict(X_i[0:4096].reshape(1, -1))[0]
            elif prob[0][3] + prob[0][4] > 0.5: # sit or ride
                pred = classifier_sit_ride.predict(X_i[4096:8192].reshape(1, -1))[0]
            elif prob[0][5] + prob[0][7] > 0.5: # snowboard, or skateboard
                pred = classifier_snow_skate.predict(X_i[4096:8192].reshape(1, -1))[0]
            else:
                pred = classifier.predict(X_i.reshape(1, -1))[0]
            score += (1 if pred == y_i else 0)
            prediction.append(pred)    
                      
        score /= len(X)
        overall_score += score
        y_pred += prediction
        y_true += y.tolist()
        #print(prediction)
        print(f'The accuracy of action {action}: {score}')
    
    print(f'↳ The overall accuracy: {overall_score / len(actions)}')

    # Calculate the confusion matrix
    print('Confusion matrix:')
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=[_ for _ in actions if _ not in skiped_actions])
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    cm_display.plot(cmap='Blues', xticks_rotation=280, ax=ax)
    plt.show()


if __name__ == '__main__':
    train_id_path, test_id_path, train_label, test_label = split_dataset()
    # train(train_id_path, train_label)
    test('0610-132034', test_id_path, test_label)
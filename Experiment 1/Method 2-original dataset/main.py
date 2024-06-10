import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from extract_feature import extract_feature
import pickle
import time
import matplotlib.pyplot as plt


actions = []

def split_dataset():
    print('\n ======== Splitting dataset ======== ')
    root_dir = "data/all"
    datas = []
    labels = []

    # Load all image id, skip the actions that have less than 400 images
    for i, action in enumerate(os.listdir(root_dir)):
        action_path = os.path.join(root_dir, action)
        image_ids = os.listdir(action_path)

        # If there are less than 400 images, ignore this image
        # Ignore list: drink, kick, point, read
        if len(image_ids) < 400:
            print(f'{action} has less than 400 images, ignore')
            continue

        # If there are enough images, add the first 400 images to the dataset
        datas += [os.path.join(action_path, image_id) for image_id in image_ids[:400]]
        labels += ([i] * 400)
        actions.append(action)
        print(f'{i} - {action}: {len(image_ids)} images')

    # Split the dataset
    train_id_path, test_id_path, train_label, test_label = train_test_split(datas, labels, test_size=0.1, random_state=42, stratify=labels)
    train_id_path, train_label = zip(*sorted(zip(train_id_path, train_label)))
    test_id_path, test_label = zip(*sorted(zip(test_id_path, test_label)))
    print('# of Train dataset:', len(train_id_path))
    print('# of Test dataset:', len(test_id_path))

    return train_id_path, test_id_path, train_label, test_label

def extract_features(type, id_paths, labels):
    print(f'Extracting features on {type}...')
    prev_action = ''
    X = [] # features
    y = [] # actions
    image_types = ['agent.jpg', 'object.jpg', 'intersection.jpg', 'union.jpg']
    action, prev_action = '', ''
    finish_extracting = True
    
    for id_path, label in zip(id_paths, labels):
        action = id_path.split('/')[2]

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
    y = []
    for action in enumerate(actions):
        X.append(np.load(f'features/all/train/{action}_X.npy'))
        y.append(np.load(f'features/all/train/{action}_y.npy'))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print('Total number of images', X.shape[0])
    
    # Train the classifiers
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                    tol=1e-2, max_iter=50000, class_weight='balanced',   # hyparameters (?)
                                    verbose=1, n_jobs=-1)

    print('\nTraining the classifier...')
    classifier.fit(X, y)

    # Save the model
    timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
    with open(f'model/model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print('\nModel saved!')
    
    # Test the classifier in training dataset
    print('Finished training. The accuracy of the model on training dataset is: ')
    overall_score = classifier.score(X, y)
    for action in actions:
        X = np.load(f'features/all/train/{action}_X.npy')
        y = np.load(f'features/all/train/{action}_y.npy')
        score = classifier.score(X, y)
        print(f'The accuracy of action {action}: {score}')
    print(f'↳ The overall accuracy: {overall_score}')


def test(model_name, test_id_path, test_label):
    print('\n ======== Testing ======== ')

    # Load images and extract features using AlexNet
    extract_features('test', test_id_path, test_label)

    # Load the model
    print('\nLoading the model...')
    with open(f'model/{model_name}', 'rb') as f:
        classifier = pickle.load(f)

    # Test the classifier in testing dataset
    print('\nLoading features...')
    X = []
    y = []
    y_pred = []
    y_true = []
    for action in actions:
        tmp_X = np.load(f'features/all//test_AlexNet//{action}_X.npy')
        tmp_y = np.load(f'features/all//test_AlexNet//{action}_y.npy')
        X.append(tmp_X)
        y.append(tmp_y)
        score = classifier.score(tmp_X, tmp_y)
        prediction = classifier.predict(tmp_X)
        y_pred += prediction.tolist()
        y_true += tmp_y.tolist()
        #print(classifier.predict(tmp_X))
        print(f'The accuracy of action {action}: {score}')
    
    # Calculate the overall accuracy
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    overall_score = classifier.score(X, y)
    print(f'↳ The overall accuracy: {overall_score}')

    print('Confusion matrix:')
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=actions)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    plt.rcParams.update({'font.size': 18})
    cm_display.plot(cmap='Blues', xticks_rotation=280, ax=ax, colorbar=False)
    plt.show()


def test_one_image(model_name):
    action = input("請輸入 action: ")
    image_id = input("請輸入 image_id: ")

    print(' ======== Testing ======== ')

    # Load the model
    model_path = f'model/{model_name}'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    image_id_path = f'data/test/{action}/{image_id}'
    if not os.path.exists(image_id_path):
        raise FileNotFoundError(f"Image path not found: {image_id_path}")
    
    image_types = ['agent.jpg', 'object.jpg', 'intersection.jpg', 'union.jpg']
    real_image_type = os.listdir(image_id_path)
    images = []
    for image_type in image_types:
        # If there is no intersection, use white image
        if image_type not in real_image_type:
            image_path = 'white.jpeg'
        else:
            image_path = os.path.join(image_id_path, image_type)
        image = Image.open(image_path)
        images.append(image)

    show_image_path = os.path.join(image_id_path, 'union.jpg')
    if os.path.exists(show_image_path):
        show_image = Image.open(show_image_path)
        plt.imshow(show_image)
        plt.title('Picture')
        plt.axis('off')
        plt.show()
    components = extract_feature(images).reshape(1, -1)
    probabilities = classifier.predict_proba(components)
    predicted_action = actions[classifier.predict(components)[0]]
    print(f"Predicted action: {predicted_action}")


if __name__ == '__main__':
    train_id_path, test_id_path, train_label, test_label = split_dataset()
    #train(train_id_path, train_label)
    test('model_0603-025850.pkl', test_id_path, test_label)
    #test_one_image('model_0603-025850.pkl')
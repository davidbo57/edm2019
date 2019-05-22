# import the necessary packages
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from metrics import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adagrad


def AES_Model():
    model = Sequential()
    model.add(Dense(96, input_shape=(96,), kernel_initializer='normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, kernel_initializer='normal'))
    opt = Adam()
    model.compile(loss="mse", optimizer=opt)
    return model


RUBRIC_1 = 0
RUBRIC_2 = 1
RUBRIC_5 = 2
RUBRIC_6 = 3

SELECTED_RUBRIC = RUBRIC_6
SCALE_LOWER_LIMIT = 2
SCALE_UPPER_LIMIT = 12
ITERATIONS = 5

EPOCHS = 300
BATCH_SIZE = 128
N_SPLITS = 5

X = np.loadtxt("../data/dataset96_weights.csv", dtype='float', delimiter=',', skiprows=1, usecols=tuple(range(0, 96)))
Y = np.loadtxt("../data/dataset96_weights.csv", dtype='float', delimiter=',', skiprows=1, usecols=(96, 97, 100, 101, 102))

# Batch normalization of raw data
means = np.mean(X, axis=0)
X = np.subtract(X, means)
variances = np.var(X, axis=0, ddof=1)
X = np.divide(X, variances)

training_kappas = []
training_losses = []
training_exact = []
training_adjacent1 = []
training_adjacent2 = []

validation_kappas = []
validation_losses = []
validation_exact = []
validation_adjacent1 = []
validation_adjacent2 = []

testing_kappas = []
testing_losses = []
testing_exact = []
testing_adjacent1 = []
testing_adjacent2 = []

training_ensemble_kappas = []
training_ensemble_losses = []
training_ensemble_exact = []
training_ensemble_adjacent1 = []
training_ensemble_adjacent2 = []

testing_ensemble_kappas = []
testing_ensemble_losses = []
testing_ensemble_exact = []
testing_ensemble_adjacent1 = []
testing_ensemble_adjacent2 = []

for i in range(ITERATIONS):
    print("Trial" + str(i))
    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.15)

    train_model = AES_Model()
    history = train_model.fit(trainX, trainY[:,0:4], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, sample_weight=trainY[:,4])

    machine_scores = train_model.predict(trainX, batch_size=BATCH_SIZE)
    machine_scores = np.rint(machine_scores)
    machine_scores = np.where(machine_scores > 12, 12, machine_scores)
    machine_scores = np.where(machine_scores < 2, 2, machine_scores)
    machine_scores = machine_scores[:,SELECTED_RUBRIC]

    human_scores = trainY[:,SELECTED_RUBRIC]

    loss = train_model.evaluate(trainX, trainY[:,0:4], sample_weight=trainY[:,4], verbose=0)
    kappa = quadratic_weighted_kappa(rater_a=machine_scores, rater_b=human_scores, min_rating=SCALE_LOWER_LIMIT, max_rating=SCALE_UPPER_LIMIT)
    exact = exact_match(rater_a=machine_scores, rater_b=human_scores)
    adjacent1 = adjacent_match(rater_a=machine_scores, rater_b=human_scores)
    adjacent2 = adjacent_match2(rater_a=machine_scores, rater_b=human_scores)
    
    training_kappas.append(kappa)
    training_exact.append(exact)
    training_adjacent1.append(adjacent1)
    training_adjacent2.append(adjacent2)
    training_losses.append(loss)

    print("Quadratic Weighted Kappa:", kappa)
    print("Exact match (%):", exact)
    print("Adjacent match (%):", adjacent1)
    print("Adjacent match2 (%):", adjacent2)
    
    # seed = 7
    # np.random.seed(seed)

    # Manual cross-validation
    kfold = KFold(n_splits=N_SPLITS, shuffle=True)
    # kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    cvscores = []
    kappas = []
    exact_matches = []
    adjacent_matches = []
    adjacent_matches2 = []
    cv_loss = np.zeros(EPOCHS, dtype=np.float32)
    train_avg_machine_scores = []
    test_avg_machine_scores = []
    for train, test in kfold.split(trainX, trainY):
        cv_model = AES_Model()
        H = cv_model.fit(trainX[train], trainY[train,0:4], validation_data=(trainX[test], trainY[test,0:4]), epochs=EPOCHS, batch_size=BATCH_SIZE, sample_weight=trainY[train,4], verbose=2)
        score = cv_model.evaluate(trainX[test], trainY[test,0:4], sample_weight=trainY[test,4], verbose=0)
        machine_scores = cv_model.predict(trainX[test], batch_size=BATCH_SIZE)
        machine_scores = np.rint(machine_scores)
        machine_scores = np.where(machine_scores > 12, 12, machine_scores)
        machine_scores = np.where(machine_scores < 2, 2, machine_scores)
        machine_scores = machine_scores[:,SELECTED_RUBRIC]
        human_scores = trainY[test,0:4]
        human_scores = human_scores[:,SELECTED_RUBRIC]

        kappa = quadratic_weighted_kappa(rater_a=machine_scores, rater_b=human_scores, min_rating=2, max_rating=12)
        ex_mat = exact_match(rater_a=machine_scores, rater_b=human_scores)
        adj_mat = adjacent_match(rater_a=machine_scores, rater_b=human_scores)
        adj_mat2 = adjacent_match2(rater_a=machine_scores, rater_b=human_scores)
        print("-", cv_model.metrics_names[0]+":", score, "-", "Quadratic Weighted Kappa:", kappa, "-",
            "Exact Match:", ex_mat, "-", "Adjacent Match:", adj_mat, "-", "Adjacent Match2:", adj_mat2)
        kappas.append(kappa)
        exact_matches.append(ex_mat)
        adjacent_matches.append(adj_mat)
        adjacent_matches2.append(adj_mat2)
        cvscores.append(score)
        cv_loss = cv_loss + H.history["val_loss"]

        train_machine_scores = cv_model.predict(trainX, batch_size=BATCH_SIZE)
        train_machine_scores = np.where(train_machine_scores > 12, 12, train_machine_scores)
        train_machine_scores = np.where(train_machine_scores < 2, 2, train_machine_scores)
        train_machine_scores = train_machine_scores[:,SELECTED_RUBRIC]
        train_avg_machine_scores.append(train_machine_scores)
        
        test_machine_scores = cv_model.predict(testX, batch_size=BATCH_SIZE)
        test_machine_scores = np.where(test_machine_scores > 12, 12, test_machine_scores)
        test_machine_scores = np.where(test_machine_scores < 2, 2, test_machine_scores)
        test_machine_scores = test_machine_scores[:,SELECTED_RUBRIC]
        test_avg_machine_scores.append(test_machine_scores)

    cv_loss = cv_loss / float(N_SPLITS)

    loss = np.mean(cvscores)
    kappa = np.mean(kappas)
    exact = np.mean(exact_matches)
    adjacent1 = np.mean(adjacent_matches)
    adjacent2 = np.mean(adjacent_matches2)

    print("Mean Loss Results:", loss, np.std(cvscores))
    print("Mean Quadratic Weighted Kappa:", kappa, np.std(kappas))
    print("Mean Exact Match:", exact, np.std(exact_matches))
    print("Mean Adjacent Match:", adjacent1, np.std(adjacent_matches))
    print("Mean Adjacent Match2:", adjacent2, np.std(adjacent_matches2))

    validation_kappas.append(kappa)
    validation_exact.append(exact)
    validation_adjacent1.append(adjacent1)
    validation_adjacent2.append(adjacent2)
    validation_losses.append(loss)

    train_avg_machine_scores = np.array(train_avg_machine_scores)
    train_avg_machine_scores = np.mean(train_avg_machine_scores, axis=0)
    train_avg_machine_scores = np.rint(train_avg_machine_scores)
    train_human_scores = trainY[:,SELECTED_RUBRIC]

    kappa = quadratic_weighted_kappa(rater_a=train_avg_machine_scores, rater_b=train_human_scores, min_rating=SCALE_LOWER_LIMIT, max_rating=SCALE_UPPER_LIMIT)
    exact = exact_match(rater_a=train_avg_machine_scores, rater_b=train_human_scores)
    adjacent1 = adjacent_match(rater_a=train_avg_machine_scores, rater_b=train_human_scores)
    adjacent2 = adjacent_match2(rater_a=train_avg_machine_scores, rater_b=train_human_scores)
    print("Ensemble - Train - Quadratic Weighted Kappa:", kappa)
    print("Ensemble - Train - Exact match (%):", exact)
    print("Ensemble - Train - Adjacent match (%):", adjacent1)
    print("Ensemble - Train - Adjacent match2 (%):", adjacent2)
    training_ensemble_kappas.append(kappa)
    training_ensemble_exact.append(exact)
    training_ensemble_adjacent1.append(adjacent1)
    training_ensemble_adjacent2.append(adjacent2)

    test_avg_machine_scores = np.array(test_avg_machine_scores)
    test_avg_machine_scores = np.mean(test_avg_machine_scores, axis=0)
    test_avg_machine_scores = np.rint(test_avg_machine_scores)
    test_human_scores = testY[:,SELECTED_RUBRIC]
    kappa = quadratic_weighted_kappa(rater_a=test_avg_machine_scores, rater_b=test_human_scores, min_rating=SCALE_LOWER_LIMIT, max_rating=SCALE_UPPER_LIMIT)
    exact = exact_match(rater_a=test_avg_machine_scores, rater_b=test_human_scores)
    adjacent1 = adjacent_match(rater_a=test_avg_machine_scores, rater_b=test_human_scores)
    adjacent2 = adjacent_match2(rater_a=test_avg_machine_scores, rater_b=test_human_scores)
    print("Ensemble - Test - Quadratic Weighted Kappa:", kappa)
    print("Ensemble - Test - Exact match (%):", exact)
    print("Ensemble - Test - Adjacent match (%):", adjacent1)
    print("Ensemble - Test - Adjacent match2 (%):", adjacent2)
    testing_ensemble_kappas.append(kappa)
    testing_ensemble_exact.append(exact)
    testing_ensemble_adjacent1.append(adjacent1)
    testing_ensemble_adjacent2.append(adjacent2)

    test_machine_scores = train_model.predict(testX, batch_size=BATCH_SIZE)
    test_machine_scores = np.rint(test_machine_scores)
    test_machine_scores = np.where(test_machine_scores > 12, 12, test_machine_scores)
    test_machine_scores = np.where(test_machine_scores < 2, 2, test_machine_scores)
    test_machine_scores = test_machine_scores[:,SELECTED_RUBRIC]
    test_score = train_model.evaluate(testX, testY, verbose=2)
    kappa = quadratic_weighted_kappa(rater_a=test_machine_scores, rater_b=test_human_scores, min_rating=SCALE_LOWER_LIMIT, max_rating=SCALE_UPPER_LIMIT)
    exact = exact_match(rater_a=test_machine_scores, rater_b=test_human_scores)
    adjacent1 = adjacent_match(rater_a=test_machine_scores, rater_b=test_human_scores)
    adjacent2 = adjacent_match2(rater_a=test_machine_scores, rater_b=test_human_scores)
    print("Test - Loss Results:", test_score)
    print("Test - Quadratic Weighted Kappa:", kappa)
    print("Test - Exact match (%):", exact)
    print("Test - Adjacent match (%):", adjacent1)
    print("Test - Adjacent match2 (%):", adjacent2)
    testing_kappas.append(kappa)
    testing_exact.append(exact)
    testing_adjacent1.append(adjacent1)
    testing_adjacent2.append(adjacent2)
    testing_losses.append(test_score)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), cv_loss, label="cv_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.ylim((0, 5))
    plt.legend()
    plt.show()

    # estimator = KerasRegressor(build_fn=AES_Model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    # plot_learning_curve(estimator, "Learning Curve", X, Y, cv=10)
    # plt.show()

print("Training mean kappa = ", np.mean(np.array(training_kappas)))
print("Training mean exact = ", np.mean(np.array(training_exact)))
print("Training mean adjacent1 = ", np.mean(np.array(training_adjacent1)))
print("Training mean adjacent2 = ", np.mean(np.array(training_adjacent2)))

print("Validation mean kappa = ", np.mean(np.array(validation_kappas)))
print("Validation mean exact = ", np.mean(np.array(validation_exact)))
print("Validation mean adjacent1 = ", np.mean(np.array(validation_adjacent1)))
print("Validation mean adjacent2 = ", np.mean(np.array(validation_adjacent2)))

print("Training ensemble mean kappa = ", np.mean(np.array(training_ensemble_kappas)))
print("Training ensemble mean exact = ", np.mean(np.array(training_ensemble_exact)))
print("Training ensemble mean adjacent1 = ", np.mean(np.array(training_ensemble_adjacent1)))
print("Training ensemble mean adjacent2 = ", np.mean(np.array(training_ensemble_adjacent2)))

print("Testing ensemble mean kappa = ", np.mean(np.array(testing_ensemble_kappas)))
print("Testing ensemble mean exact = ", np.mean(np.array(testing_ensemble_exact)))
print("Testing ensemble mean adjacent1 = ", np.mean(np.array(testing_ensemble_adjacent1)))
print("Testing ensemble mean adjacent2 = ", np.mean(np.array(testing_ensemble_adjacent2)))

print("Testing mean kappa = ", np.mean(np.array(testing_kappas)))
print("Testing mean exact = ", np.mean(np.array(testing_exact)))
print("Testing mean adjacent1 = ", np.mean(np.array(testing_adjacent1)))
print("Testing mean adjacent2 = ", np.mean(np.array(testing_adjacent2)))

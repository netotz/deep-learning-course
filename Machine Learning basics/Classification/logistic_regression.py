from templates.classifier import Classifier

# instance of the ML classifier template
classifier = Classifier("Machine Learning basics/Classification/Social_Network_Ads.csv")

# strings for axes labels
xlabel = "Age"
ylabel = "Estimated salary"

# plot the training results
classifier.plotPrediction(classifier.X_train, classifier.y_train, "Training", xlabel, ylabel)
# plot the testing results
classifier.plotPrediction(classifier.X_test, classifier.y_test, "Testing", xlabel, ylabel)
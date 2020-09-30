%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
plotCase(X,D)
bestK = 0;
bestAcc = 0;

for y = 1:50

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
 %XTrain = XBins{1};
 %LTrain = LBins{1};
 
 XTrain1 = combineBins(XBins, [1,2]);
 LTrain1 = combineBins(LBins, [1,2]);
 
 XTest1  = XBins{3};
 LTest1  = LBins{3};
 
 XTrain2 = combineBins(XBins, [1,3]);
 LTrain2 = combineBins(LBins, [1,3]);
 
 XTest2  = XBins{2};
 LTest2  = LBins{2};
 
 XTrain3 = combineBins(XBins, [2,3]);
 LTrain3 = combineBins(LBins, [2,3]);
 
 XTest3  = XBins{1};
 LTest3  = LBins{1};
 

 

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors
k = y;



% Classify training data
LPredTrain1 = kNN(XTrain1, k, XTrain1, LTrain1);
% Classify test data
LPredTest1  = kNN(XTest1 , k, XTrain1, LTrain1);

LPredTrain2 = kNN(XTrain2, k, XTrain2, LTrain2);
% Classify test data
LPredTest2  = kNN(XTest2 , k, XTrain2, LTrain2);

LPredTrain3 = kNN(XTrain3, k, XTrain3, LTrain3);
% Classify test data
LPredTest3  = kNN(XTest3 , k, XTrain3, LTrain3);


%% Calculate The Confusion Matrix and the Accuracy
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

% The confucionMatrix
cM1 = calcConfusionMatrix(LPredTest1, LTest1);

% The accuracy
acc1 = calcAccuracy(cM1);

% The confucionMatrix
cM2 = calcConfusionMatrix(LPredTest2, LTest2);

% The accuracy
acc2 = calcAccuracy(cM2);

% The confucionMatrix
cM3 = calcConfusionMatrix(LPredTest3, LTest3);

% The accuracy
acc3 = calcAccuracy(cM3);

acc = (acc1 + acc2 + acc3)/3;

if acc > bestAcc
    bestK = y;
    bestAcc = acc;
end
end

bestK
bestAcc

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain1, LTrain1, LPredTrain1, XTest1, LTest1, LPredTest1, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end

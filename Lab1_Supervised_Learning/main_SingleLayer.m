%% This script will help you test your single layer neural network code

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

%% Select a subset of the training samples

numBins = 3;                    % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
 XTrain = XBins{1,2};
 DTrain = DBins{1,2};
 LTrain = LBins{1,2};
 XTest  = XBins{3};
 DTest  = DBins{3};
 LTest  = LBins{3};
 
 %{
 XTrain1 = combineBins(XBins, [1,2]);
 LTrain1 = combineBins(LBins, [1,2]);
 DTrain1 = combineBins(LBins, [1,2]);
 
 XTest1  = XBins{3};
 LTest1  = LBins{3};
 DTest1  = DBins{3};
 
 XTrain2 = combineBins(XBins, [1,3]);
 LTrain2 = combineBins(LBins, [1,3]);
 DTrain2 = combineBins(LBins, [1,3]);
 
 XTest2  = XBins{2};
 LTest2  = LBins{2};
 DTest2  = DBins{2};
 
 XTrain3 = combineBins(XBins, [2,3]);
 LTrain3 = combineBins(LBins, [2,3]);
 DTrain3 = combineBins(LBins, [2,3]);
 
 XTest3  = XBins{1};
 LTest3  = LBins{1};
 DTest3  = DBins{1};
 
 %}

%% Modify the X Matrices so that a bias is added
%  Note that the bias must be the last feature for the plot code to work

bias =  ones(size(XTrain,1),1 );

% The training data
 XTrain = [XTrain bias];

% The test data
 XTest = [XTest bias];
 
 %{
 % The training data
 XTrain1 = [XTrain1 bias];

% The test data
 XTest1 = [XTest1 bias];
 
 % The training data
 XTrain2 = [XTrain2 bias];

% The test data
 XTest2 = [XTest2 bias];
 
 % The training data
 XTrain3 = [XTrain3 bias];

% The test data
 XTest3 = [XTest3 bias];
 %}

%% Train your single layer network
%  Note: You need to modify trainSingleLayer() and runSingleLayer()
%  in order to train the network

numIterations = 200000;  % Change this, number of iterations (epochs)
learningRate  = 0.0001; % Change this, your learning rate
%W0 = 0; % Change this, initialize your weight matrix W
classes = unique(L);

W0 = randn( size(XTrain,2), size(classes,1) ); 


% Run training loop
tic;
[W, ErrTrain, ErrTest] = trainSingleLayer(XTrain, DTrain, XTest, DTest, W0, numIterations, learningRate);
trainingTime = toc;

%{

tic;
[W1, ErrTrain, ErrTest] = trainSingleLayer(XTrain1, DTrain1, XTest1, DTest1, W0, numIterations, learningRate);
[W2, ErrTrain, ErrTest] = trainSingleLayer(XTrain2, DTrain2, XTest2, DTest2, W0, numIterations, learningRate);
[W3, ErrTrain, ErrTest] = trainSingleLayer(XTrain3, DTrain3, XTest3, DTest3, W0, numIterations, learningRate);
trainingTime = toc;



%}

%% Plot errors
%  Note: You should not have to modify this code

[minErrTest, minErrTestInd] = min(ErrTest);

figure(1101);
clf;
semilogy(ErrTrain, 'k', 'linewidth', 1.5);
hold on;
semilogy(ErrTest, 'r', 'linewidth', 1.5);
semilogy(minErrTestInd, minErrTest, 'bo', 'linewidth', 1.5);
hold off;
xlim([0,numIterations]);
grid on;
title('Training and Test Errors, Single Layer');
legend('Training Error', 'Test Error', 'Min Test Error');
xlabel('Epochs');
ylabel('Error');

%% Calculate the Confusion Matrix and the Accuracy of the data
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

tic;
[~, LPredTrain] = runSingleLayer(XTrain, W);
[~, LPredTest ] = runSingleLayer(XTest , W);
classificationTime = toc/(length(XTest) + length(XTrain));

% The confucionMatrix
cM = calcConfusionMatrix(LPredTest, LTest)

% The accuracy
acc = calcAccuracy(cM);

%{
tic;
[~, LPredTrain1] = runSingleLayer(XTrain1, W1);
[~, LPredTest1 ] = runSingleLayer(XTest1 , W1);
classificationTime = toc/(length(XTest1) + length(XTrain1));

% The confucionMatrix
cM1 = calcConfusionMatrix(LPredTest1, LTest1)

% The accuracy
acc1 = calcAccuracy(cM1);

tic;
[~, LPredTrain2] = runSingleLayer(X2rain2, W2);
[~, LPredTest2 ] = runSingleLayer(XTest2 , W2);
classificationTime = toc/(length(XTest2) + length(XTrain2));

% The confucionMatrix
cM2 = calcConfusionMatrix(LPredTest2, LTest2)

% The accuracy
acc2 = calcAccuracy(cM2);

tic;
[~, LPredTrain3] = runSingleLayer(XTrain3, W3);
[~, LPredTest3 ] = runSingleLayer(XTest3 , W3);
classificationTime = toc/(length(XTest3) + length(XTrain3));

% The confucionMatrix
cM3 = calcConfusionMatrix(LPredTest3, LTest3)

% The accuracy
acc3 = calcAccuracy(cM3);

acc = (acc1 + acc2 + acc3)/3

%}



disp(['Time spent training: ' num2str(trainingTime) ' sec']);
disp(['Time spent classifying 1 sample: ' num2str(classificationTime) ' sec']);
disp(['Test accuracy: ' num2str(acc)]);

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'single', {W}, []);
else
    plotResultsOCR(XTest, LTest, LPredTest);
end

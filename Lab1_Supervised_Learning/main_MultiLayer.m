%% This script will help you test your multi-layer neural network code

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

%% Select a subset of the training features

numBins = 3;                    % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select features at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here

 %{
 XTrain = XBins{1,2};
 DTrain = DBins{1,2};
 LTrain = LBins{1,2};
 XTest  = XBins{3};
 DTest  = DBins{3};
 LTest  = LBins{3};
 %}
 
 %{
 XTrain = combineBins(XBins, [1,2,3]);
 DTrain = combineBins(DBins, [1,2,3]);
 LTrain = combineBins(LBins, [1,2,3]);
 
 XTest  = combineBins(XBins, [4,5]);
 DTest  = combineBins(DBins, [4,5]);
 LTest  = combineBins(LBins, [4,5]);
 %}
 
%{ 
XTrain = X(332:667,:);
DTrain = D(332:667,:);
LTrain = L(332:667,:);
%}

 XTrain1 = combineBins(XBins, [1,2]);
 DTrain1 = combineBins(DBins, [1,2]);
 LTrain1 = combineBins(LBins, [1,2]);
 
 XTest1  = XBins{3};
 LTest1  = LBins{3};
 DTest1  = DBins{3};
 
 
 XTrain2 = combineBins(XBins, [1,3]);
 DTrain2 = combineBins(DBins, [1,3]);
 LTrain2 = combineBins(LBins, [1,3]);
 
 XTest2  = XBins{2};
 LTest2  = LBins{2};
 DTest2  = DBins{2};
 
 XTrain3 = combineBins(XBins, [2,3]);
 DTrain3 = combineBins(DBins, [2,3]);
 LTrain3 = combineBins(LBins, [2,3]);
 
 XTest3  = XBins{1};
 DTest3  = DBins{1};
 LTest3  = LBins{1};
%% Modify the X Matrices so that a bias is added

bias =  ones(size(XTrain,1),1 );
biastest =  ones(size(XTest,1),1 );



% The training data
 XTrain1 = [XTrain1 bias];
 XTrain2 = [XTrain2 bias];
 XTrain3 = [XTrain3 bias];

% The test data
 XTest1 = [XTest1 biastest];
 XTest2 = [XTest2 biastest];
 XTest3 = [XTest3 biastest];

%% Train your multi-layer network
%  Note: You need to modify trainMultiLayer() and runMultiLayer()
%  in order to train the network

numHidden     = 10;     % Change this, number of hidden neurons 
numIterations = 10000;   % Change this, number of iterations (epochs)
learningRate  = 0.005; % Change this, your learning rate
% W0 = 0; % Initialize your weight matrix W
classes = unique(L);

W0 = randn( size(XTrain,2), numHidden ) / 100;

%V0 = 0; % Initialize your weight matrix V

V0 = randn( (numHidden+1) , size(classes,1) ) / 100;

%runMultiLayer(XTrain(1,:), W0, V0)


% Run training loop
tic;
[W,V,ErrTrain,ErrTest] = trainMultiLayer(XTrain, DTrain, XTest, DTest ,W0, V0, numIterations, learningRate);
trainingTime = toc;

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
title('Training and Test Errors, Multi-layer');
legend('Training Error', 'Test Error', 'Min Test Error');
xlabel('Epochs');
ylabel('Error');

%% Calculate the Confusion Matrix and the Accuracy of the data
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

tic;
[~, LPredTrain] = runMultiLayer(XTrain, W, V);
[~, LPredTest ] = runMultiLayer(XTest , W, V);
classificationTime = toc/(length(XTest) + length(XTrain));

% The confucionMatrix
cM = calcConfusionMatrix(LPredTest, LTest)

% The accuracy
acc = calcAccuracy(cM);

disp(['Time spent training: ' num2str(trainingTime) ' sec']);
disp(['Time spent classifying 1 sample: ' num2str(classificationTime) ' sec']);
disp(['Test accuracy: ' num2str(acc)]);

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'multi', {W,V}, []);
else
    plotResultsOCR(XTest, LTest, LPredTest);
end

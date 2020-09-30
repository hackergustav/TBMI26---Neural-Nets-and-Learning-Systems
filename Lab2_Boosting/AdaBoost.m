%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 300;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 150;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
tic;

D = ones(1 ,size(xTrain,2))/size(xTrain,2);
Dhist = ones(nbrWeakClassifiers ,size(xTrain,2))/size(xTrain,2);

WC = ones(nbrWeakClassifiers,3);

Alpha = ones(nbrWeakClassifiers,1);

Emin = ones(nbrWeakClassifiers,1);

for i = 1:nbrWeakClassifiers  
    
    for k = 1:nbrHaarFeatures
    
            
        
        for j = 1:nbrTrainImages
            p = 1;
            C = WeakClassifier(xTrain(k,j), p, xTrain(k,:));
            E = WeakClassifierError(C,D,yTrain);
                       
            if E > 0.5
               % if E > 1
               %     p = p*(-1);
               %     E = 0.00000000000000001;
               % else
                    p = p*(-1); 
                    E = 1-E;
               % end
            end
            
            if E < Emin(i)
                Emin(i) = E;
                WC(i,1) = xTrain(k,j);
                WC(i,2) = p;
                WC(i,3) = k;
            end 

        end
    end
    
    if Emin(i) == 0.5
        break;
    end
    
    Alpha(i) = log((1 - Emin(i)) / Emin(i)) / 2;
    
    htrain = (WeakClassifier(WC(i,1),WC(i,2),xTrain(WC(i,3),:)));
    
    Dhist(i,:) = D;
    
    D = D.*exp(-Alpha(i) * yTrain .* htrain );
    D(D>0.01) = 0.01; % Avoiding big weights for outliers, maybe too big
    %D(D<0.0001) = 0.0001; %Avoiding unstable behavior
    D = D./sum(D);
    
end
trainingTime = toc

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

h = zeros(nbrWeakClassifiers,length(xTrain));

 for i = 1:nbrWeakClassifiers
    h(i,:) = Alpha(i)*WeakClassifier(WC(i,1),WC(i,2), xTrain(WC(i,3),:));
    
 end
 
 H = sign(sum(h));
 
 trainAcc = sum(yTrain == H)/length(yTrain)
 %------------------------------------------------

h = zeros(nbrWeakClassifiers,length(xTest));

 for i = 1:nbrWeakClassifiers
         
    h(i,:) = Alpha(i)*WeakClassifier(WC(i,1),WC(i,2), xTest(WC(i,3),:));
   
 end
 
 H = sign(sum(h));
 
 testAcc = sum(yTest == H)/length(yTest)
 
 
 

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

Acc_test = zeros(nbrWeakClassifiers,1);

Acc_train = zeros(nbrWeakClassifiers,1);

for k = 1:nbrWeakClassifiers
    
     h_test = zeros(k,length(xTest));
     
     h_train = zeros(k,length(xTrain));
     
     sortedAlpha = sort(Alpha,1,'descend');

     for i = 1:k
        index = find(Alpha == sortedAlpha(i),1,'first') ;
     
        h_test(i,:) = Alpha(index)*WeakClassifier(WC(index,1),WC(index,2), xTest(WC(index,3),:));
          
     end
     
      for i = 1:k
        index = find(Alpha == sortedAlpha(i),1,'first') ;
            
        h_train(i,:) = Alpha(index)*WeakClassifier(WC(index,1),WC(index,2), xTrain(WC(index,3),:));
   
     end

     H_test = sign(sum(h_test));
     
     H_train = sign(sum(h_train));

     Acc_test(k) = sum(yTest == H_test)/length(yTest);
     Acc_train(k) = sum(yTrain == H_train)/length(yTrain);
end

xAxis = 1:nbrWeakClassifiers;
figure(4);
plot(xAxis,Acc_test, 'b')
hold on
plot(xAxis,Acc_train,'r') 
hold off 

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

wrong = H - yTest;
count = 1;

figure(5);
colormap gray;
for k=1:length(wrong)
    
    if wrong(k) == 2
   
    subplot(5,5,count), imagesc(testImages(:,:,k));
    count = count + 1;
    axis image;
    axis off;
    end
    
    if count > 25
        break;
    end
end

count = 1;
figure(6);
colormap gray;
for k=1:length(wrong)
    
    if wrong(k) == -2
    
    subplot(5,5,count), imagesc(testImages(:,:,k));
    count = count + 1;
    axis image;
    axis off;
    end
    
    if count > 25
        break;
    end
end



%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(7);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,WC(k,3)),[-1 2]);
    axis image;
    axis off;
end

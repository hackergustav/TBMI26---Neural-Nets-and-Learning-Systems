function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);
distances = zeros(size(XTrain,1),1);
kpoints = zeros(k,1);
count = zeros(NClasses,1);

for i = 1:length(X(:,1))
    
    count = zeros(NClasses,1);
    
    for j = 1:length(XTrain(:,1))
        distances(j) = norm(X(i,:) - XTrain(j,:));
    end
    
    for h = 1:k
        closest = min(distances);        
        kpoints(h) = find(closest == distances,1);
        distances(kpoints(h)) = inf; 
    end
    
    
    for h =1:k
        count(find(LTrain(kpoints(h)) == classes)) = count(find(LTrain(kpoints(h)) == classes))+1;
    end
    
    if length(find(max(count) == count)) > 1
       
        classesInDraw = classes(find(max(count) == count));
        
       for h = 1:k
            if ismember(LTrain(kpoints(h)), classesInDraw)
                LPred(i) = LTrain(kpoints(h));
                break;
            end
       end
        
    else
        LPred(i) = classes( find(max(count) == count)    );
    end
end


end


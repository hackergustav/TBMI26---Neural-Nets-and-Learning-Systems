

h = zeros(40,length(xTrain));

for i = 1:40
    h(i,:) = Alpha(i)*WeakClassifier(Strongman(i,1),Strongman(i,2), xTrain(Strongman(i,3),:));
    
end
 
H = sign(sum(h));
 
 trainAcc = sum(yTrain == H)/length(yTrain)
 %------------------------------------------------

h = zeros(40,length(xTest));

 for i = 1:40
         
    h(i,:) = Alpha(i)*WeakClassifier(Strongman(i,1),Strongman(i,2), xTest(Strongman(i,3),:));
   
 end
 
 H = sign(sum(h));
 
 testAcc = sum(yTest == H)/length(yTest)
clear;

% Load Breast Cancer Dataset
load cancer_dataset;
% Define Inputs and Outputs
x = cancerInputs;
t = cancerTargets;

% Define the Training Functin
trainFcn = 'trainscg'; % using "trainscg" Train Function
% Define nodes(hidden layers) and epochs to use in analysis
nodes = 8; % Optimal nodes (hidden layers) from exp1)
trainEpochs = [4,8,16]; % Optimal epochs from exp1)

xLength = length(x);
classifiersNum = 3:2:25;

for e = trainEpochs
    for classifiersIndexList = 1:length(classifiersNum)
        numNN = 30;
        for i = 1:numNN
            randPermx = randperm(xLength);
            randIndTest = randPermx(1:length(randPermx)/2);
            randIndTrain = randPermx(length(randIndTest)+1 : length(randPermx));
            trainT = [];
            testT = [];
            trainY = [];
            testY = [];

            for classifiersIndex = 1:classifiersNum(classifiersIndexList)
                % Define the Neural Network
                net = patternnet(nodes, trainFcn);
                net.trainParam.epochs = e;

                % Choose Input and Output Pre/Post-Processing Functions
                net.input.processFcns = {'removeconstantrows','mapminmax'};

                %Setup Division of Data for Training, Testing
                net.divideFcn = 'divideind'; % Divide data by Indicies
                net.divideMode = 'sample';  % Divide up every sample
                net.divideParam.trainInd = randIndTrain;
                net.divideParam.testInd = randIndTest;

                % Choose a Performance Function
                net.performFcn = 'crossentropy';  % Cross-Entropy

                % Choose Confusion Plot Function
                net.plotFcns = {'plotconfusion'};   

                % Train the net
                [net,tr] = train(net, x, t);

                % Test network
                y = net(x);

                % Change the vectors to index
                tind = vec2ind(t);
                yind = vec2ind(y);

                trainT = [trainT; tind(tr.trainInd)];
                testT = [testT; tind(tr.testInd)];
                trainY = [trainY; yind(tr.trainInd)];
                testY = [testY; yind(tr.testInd)]; 
            end
            % Calculate majority with mode function
            ttrainmajority = mode(trainT);
            ttestmajority = mode(testT);
            ytrainmajority = mode(trainY);
            ytestmajority = mode(testY);
            % Calculate Error Rate percentage for Train and Test
            percentErrorsTrain(i) = sum(ttrainmajority ~= ytrainmajority)/numel(ttrainmajority); 
            percentErrorsTest(i) = sum(ttestmajority ~= ytestmajority)/numel(ttestmajority);
        end
        percentErrorsAverageTrain(classifiersIndexList) = mean(percentErrorsTrain);
        percentErrorsStdTrain(classifiersIndexList) = std(percentErrorsTrain);

        percentErrorsAverageTest(classifiersIndexList) = mean(percentErrorsTest);
        percentErrorsStdTest(classifiersIndexList) = std(percentErrorsTest);
    end
testErrorGraph = figure;
set(0, 'CurrentFigure', testErrorGraph);
test_legend_str = 'Test Error Rate';
plot(classifiersNum, percentErrorsAverageTest, 'DisplayName', test_legend_str);
title(sprintf('Ensembled Classifiers Test Error Rates with %d Epochs',e));
legend show;
xlabel('Number of Classifiers');
ylabel('Error Rate');        
end


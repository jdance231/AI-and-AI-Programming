clear;

% Load Breast Cancer Dataset
load cancer_dataset;
% Define Inputs and Outputs
x = cancerInputs;
t = cancerTargets;

% Define the Training Functin
trainFcn = 'trainscg'; % using "trainscg" Train Function
% Define nodes(hidden layers) and epochs to use in analysis
nodes = [2,8,32];
trainEpochs = [1,2,4,8,16,32,64];
% Create titles for output table
Statistics = ["Nodes","Epoch","AvgErrorTrain","AvgErrorTest","StdErrorTrain","StdErrorTest"];

for n = nodes
    fprintf('\n%d Hidden Layers:\n',n)
    for e = trainEpochs
        fprintf('\n%d Epochs:\n',e)
        % Define the Neural Network, Epochs and Hidden layers
        hiddenLayerSize = n;
        net = patternnet(hiddenLayerSize, trainFcn);
        net.trainParam.epochs = e;

        % Choose Input and Output Pre/Post-Processing Functions
        net.input.processFcns = {'removeconstantrows','mapminmax'};

        %Setup Division of Data for Training, Testing
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 50/100;
        net.divideParam.valRatio = 0/100;
        net.divideParam.testRatio = 50/100;
        
        % Choose Confusion Plot Function
        net.plotFcns = {'plotconfusion'};
    
        % Train 30 neural networks
        numNN = 30;
        nets = cell(1, numNN);
        trs = cell(1, numNN);
        for i = 1:numNN
            % Train the nets
            fprintf('Training %d/%d\n', i, numNN)
            [nets{i},trs{i}] = train(net, x, t);
            
            % Calculate the predicted classification
            y = nets{i}(x);
            % Calculate Train and Test Targets
            trainTargets = t .* trs{i}.trainMask{1};
            testTargets = t .* trs{i}.testMask{1};
            trainTargetsy = y .* trs{i}.trainMask{1};
            testTargetsy = y .* trs{i}.testMask{1};
            % Change the vectors to index
            trainind = vec2ind(trainTargets);
            testind = vec2ind(testTargets);
            trainindy = vec2ind(trainTargetsy);
            testindy = vec2ind(testTargetsy);
            % Calculate Error Rate percentage for Train and Test
            percentErrorsTrain{i} = sum(trainind ~= trainindy)/numel(trainind);
            percentErrorsTest{i} = sum(testind ~= testindy)/numel(testind);            
        end        
        % Change format of Error Rate Cell to double
        percentErrorsTrain2 = cell2mat(percentErrorsTrain);
        percentErrorsTest2 = cell2mat(percentErrorsTest);
        % Calculate the Average Error Rate
        percentErrorsAverageTrain = mean(percentErrorsTrain2);
        percentErrorsAverageTest = mean(percentErrorsTest2);
        fprintf('\nAverage Error Rate for Train: %4.5f\n', percentErrorsAverageTrain)
        fprintf('Average Error Rate for Test: %4.5f\n', percentErrorsAverageTest)
        % Calculate the Standard Deviation of Error Rate
        percentErrorsStdTrain = std(percentErrorsTrain2);
        percentErrorsStdTest = std(percentErrorsTest2);
        fprintf('\nStandard Deviation of Error rate for Train: %4.5f\n', percentErrorsStdTrain)
        fprintf('Standard Deviation of Error rate for Test: %4.5f\n', percentErrorsStdTest)
        % Add Average and Standard Deviation of Error rate to Statistics
        % table
        statadd = [n,e,percentErrorsAverageTrain,percentErrorsAverageTest,percentErrorsStdTrain,percentErrorsStdTest];
        Statistics(end+1,:) = statadd;
    end
end

% Convert to double format
StatisticsNum = str2double(Statistics([2:end],:))

% Optimal Value for Test Error Rate and Associated node/epoch values
[row, column] = min(StatisticsNum([1:end],4));
Optimal = Statistics([1,column+1],:)

%Plots
%Confusion Matrix
figure, plotconfusion(t,y)
% Line Graph
% Train Error Rate
figure; plot(StatisticsNum([1:7],2),StatisticsNum([1:7],3))
hold on
plot(StatisticsNum([8:14],2),StatisticsNum([8:14],3))
plot(StatisticsNum([15:21],2),StatisticsNum([15:21],3))
hold off
title('Train Error Rate'), xlabel('Epochs'),ylabel('Error Rate')
legend('Train Error Rate - 2 Nodes','Train Error Rate - 8 Nodes','Train Error Rate - 32 Nodes')
% Test Error Rate
figure; plot(StatisticsNum([1:7],2),StatisticsNum([1:7],4))
hold on
plot(StatisticsNum([8:14],2),StatisticsNum([8:14],4))
plot(StatisticsNum([15:21],2),StatisticsNum([15:21],4))
hold off
title('Test Error Rate'), xlabel('Epochs'),ylabel('Error Rate')
legend('Test Error Rate - 2 Nodes','Test Error Rate - 8 Nodes','Test Error Rate - 32 Nodes')
% Train Standard Deviation
figure; plot(StatisticsNum([1:7],2),StatisticsNum([1:7],5))
hold on
plot(StatisticsNum([8:14],2),StatisticsNum([8:14],5))
plot(StatisticsNum([15:21],2),StatisticsNum([15:21],5))
hold off
title('Train Standard Deviation'), xlabel('Epochs'),ylabel('Standard Deviation')
legend('Train Standard Deviation - 2 Nodes','Train Standard Deviation - 8 Nodes','Train Standard Deviation - 32 Nodes')
% Test Standard Deviation
figure; plot(StatisticsNum([1:7],2),StatisticsNum([1:7],6))
hold on
plot(StatisticsNum([8:14],2),StatisticsNum([8:14],6))
plot(StatisticsNum([15:21],2),StatisticsNum([15:21],6))
hold off
title('Test Standard Deviation'), xlabel('Epochs'),ylabel('Standard Deviation')
legend('Test Standard Deviation - 2 Nodes','Test Standard Deviation - 8 Nodes','Test Standard Deviation - 32 Nodes')
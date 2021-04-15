
%% Parameters
% clear
% clc
MNIST_images = loadMNISTImages('train-images.idx3-ubyte');
MNIST_labels = loadMNISTLabels('train-labels.idx1-ubyte');
MNIST_tests = loadMNISTImages('t10k-images.idx3-ubyte');
MNIST_test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
Layers = [256, 32, 16, 10]; % [NumberOfInputs HiddenLayer NumberOfOutputs]
LearningRate = 0.1;
TrainingCount = 10*6e4;
TestCount = 1e4;

%% Initialize

NumberOfTrainingExamples = length(MNIST_labels);

w2 = (rand(Layers(2),Layers(1))-0.5); % Get random weights
w1 = (rand(Layers(3),Layers(2))-0.5);
w0 = (rand(Layers(4),Layers(3))-0.5);

b2 = (rand(Layers(2),1));
b1 = (rand(Layers(3),1));
b0 = (rand(Layers(4),1));

x_in = MNIST_images;
x_test = MNIST_tests;

outputs = zeros(Layers(end),length(MNIST_labels));
for i=1:length(MNIST_labels)
    outputs(1+MNIST_labels(i),i) = 1;
end

test_outputs = zeros(Layers(end),length(MNIST_test_labels));
for i=1:length(MNIST_test_labels)
    test_outputs(1+MNIST_test_labels(i),i) = 1;
end

fprintf('Initialization is done.\n')
fprintf('--------------------------\n')

%% Train

% wrongs = zeros(1,Layers(end));
wrongs = 0;
k = 0;
for K=1:TrainingCount
    
    LR = 0.1; % LearningRate*(TrainingCount-K)/TrainingCount;
    k = k + 10;
    if k>6e4
        k = 1;
    end
    
    % Prediction ------------------------------------------
    a2 = sigmoid(w2*x_in(:,k) + b2);
    a1 = sigmoid(w1*a2 + b1);
    predictions = sigmoid(w0*a1 + b0);
    
    % Parameters ------------------------------------------
    % cost = (error.^2)/2;
    error = outputs(:,k) - predictions;
    
    % da(L)/dz(L) = a(L)*(1-a(L))
    da0_dz0 = predictions.*(1-predictions);
    
    % dz(L)/dw(L) = a(L-1)
    dz0_dw0 = a1; % 
    
    % dz(L)/da(L-1) = w(L)
    dz0_da1 = w0;
    
    % da(L-1)/dz(L-1) = a(L-1)*(1-a(L-1))
    da1_dz1 = a1.*(1-a1);
    
    % dz(L-1)/dw(L-1) = a(L-2)
    dz1_dw1 = a2; %
    
    % dz(L-1)/da(L-2) = w(L-1)
    dz1_da2 = w1;
    
    % da(L-2)/dz(L-2) = a(L-2)*(1-a(L-2))
    da2_dz2 = a2.*(1-a2);
    
    % dz(L-2)/dw(L-2) = a(L-3)
    dz2_dw2 = x_in(:,k);
    
    % Summations 
    % dC/da(L) = error
    dC_da0 = error;
    
    % dC/da(L-1)
    dC_da1 = (dC_da0.*da0_dz0).*dz0_da1;
    dC_da1 = sum(dC_da1)'/Layers(4);
    
    % dC/da(L-2)
    dC_da2 = (dC_da1.*da1_dz1).*dz1_da2;
    dC_da2 = sum(dC_da2)'/Layers(3);
    
    % Backpropagation ---------------------------------------
    % w0
    dC_dw0 = (dC_da0.*da0_dz0)*dz0_dw0';
    w0 = w0 + dC_dw0*LR;
    % b0
    dC_db0 = (dC_da0.*da0_dz0);
    b0 = b0 + dC_db0*LR;
    
    % w1
    dC_dw1 = (dC_da1.*da1_dz1)*dz1_dw1';
    w1 = w1 + dC_dw1*LR;
    % b1
    dC_db1 = (dC_da1.*da1_dz1);
    b1 = b1 + dC_db1*LR;
    
    % w2
    dC_dw2 = (dC_da2.*da2_dz2)*dz2_dw2';
    w1 = w1 + dC_dw1*LR;
    % b2
    dC_db2 = (dC_da2.*da2_dz2);
    b2 = b2 + dC_db2*LR;
    
    % Display
%     incorrect = false;
    if predictions(find(outputs(:,k)==1))<predictions(find(predictions==max(predictions)))
%         incorrect = true;
%         wrongs(find(outputs(:,k)==1)) = wrongs(find(outputs(:,k)==1))+1;
        wrongs = wrongs+1;
    end
%     fprintf('\nTest: %d --> Mistakes: %d --> Accuracy: %2.15f\n',K,sum(wrongs), 1-(sum(wrongs)/K));
    fprintf('\nTest: %d --> Mistakes: %d --> Accuracy: %2.8f\n',K,wrongs, 1-(wrongs/K));
%     for n=1:Layers(3)
%         fprintf(' [%d]-%4d\t|', n-1, wrongs(n));
%     end
%     fprintf('\n')
%     for n=1:Layers(3)
%         fprintf('\t%2.2f-%d\t|', predictions(n), outputs(n,k));
%     end
%     
%     figure(1)
%     X_in = reshape(MNIST_images(:,k),[16 16]);
%     figureWin = imshow(X_in);
% %     pause(0.5);
%     
%     if incorrect
%         fprintf('\t*');
%     end
end
fprintf('\n');

%% Predict after training

wrongs = zeros(1,Layers(end));
for k=1:TestCount
    
    % Prediction ------------------------------------------
    a2 = sigmoid(w2*x_test(:,k) + b2);
    a1 = sigmoid(w1*a2 + b1);
    predictions = sigmoid(w0*a1 + b0);
    
    
    % Display
    incorrect = false;
    if predictions(find(test_outputs(:,k)==1))<predictions(find(predictions==max(predictions)))
        incorrect = true;
        wrongs(find(test_outputs(:,k)==1)) = wrongs(find(test_outputs(:,k)==1))+1;
    end
    fprintf('\nTest: %d --> Mistakes: %d --> Accuracy: %2.15f\n',k,sum(wrongs), 1-(sum(wrongs)/k));
    
    for n=1:Layers(end)
        fprintf(' [%d]-%4d\t|', n-1, wrongs(n));
    end
    fprintf('\n')
    for n=1:Layers(end)
        fprintf('\t%2.2f-%d\t|', predictions(n), test_outputs(n,k));
    end
    
    
    if incorrect
        fprintf('\t*');
        figure(1)
        X_test = reshape(MNIST_tests(:,k),[16 16]);
        figureWin = imshow(X_test);
        title(find(predictions==max(predictions)))
        pause(0.5);
    end
end
fprintf('\n');


%% ---------------------------------
% Function Definitions

function y = sigmoid(x)
y = 1./(1+exp(-x));
end

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
size(images)
images = imresize(images,[16 16]);
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end

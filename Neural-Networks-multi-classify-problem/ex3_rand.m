% ex3_rand.m (is a modified version of ex3.m to scramble pixels/features)
%
%% Machine Learning Online Class - Exercise 3 | Randomize Features

%% Initialization
clear; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400; % 20x20 Input Images of Digits
num_labels = 10;         % 10 labels, from 1 to 10   
                         % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m, 100);
sel = X(rand_indices,:);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Added to randomize features (to probe that is irrelevant)
fprintf('\nRandomizing columns...\n');
X_rand = X(:, randperm(size(X,2)));

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X_rand, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X_rand);

fprintf('\nTraining Set Accuracy:%f\n', mean(double(pred == y)) * 100);

%% ============ Part 4: Predict Random Samples ============
%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
   % Display 
    fprintf('\nDisplaying Example Randomized Image\n');
    displayData(X_rand(rp(i),:));

    pred = predictOneVsAll(all_theta, X_rand(rp(i),:));
    fprintf('\nNeural Network Prediction:%d (label%d)\n', pred, y(rp(i)));

   % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

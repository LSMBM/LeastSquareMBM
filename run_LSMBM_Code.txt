%% Least Square Moment Balanced Machine (LSMBM):
% A New Approach To Estimating Cost To Completion For Construction Projects
clc;
clear all;

% Define the number of folds for k-fold cross-validation
kfold=10;

% Define grid search hyperparameter
gam =  [0.01, 0.1, 1, 10, 100];
sig =  [0.01, 0.1, 1, 10, 100];

% Prepare to store best parameters for each fold
bestParams = cell(kfold, 3);

% Process each fold
for i = 1:kfold
    % Load training and testing data for the current fold
    % Training and testing data has been partitioned into 10 fold using the crossfold_generator function
    train = readmatrix(['training_RC', num2str(i), '.xlsx']);
    test = readmatrix(['testing_RC', num2str(i), '.xlsx']);

    % Define input (features) and output (target) variables
    xtr=train(:,1:end-1);   % Training features
    ytr=train(:,end);          % Training target
    xte=test(:,1:end-1);    % Testing features
    yte=test(:,end);           % Testing target

    disp(['Processing Fold ', num2str(i)]);

    % Compute predictions and derive weights using BPNN
    [ytrp, ytep, ytr, yte, options]=simBPNN(train,test);

    % Calculate model errors on the training and testing set
    errors=postLSIM(ytrp,ytr,ytep,yte);

    % Compute case weights based on prediction errors
    w_bpnn = (1./(abs(ytrp-ytr)./ytr));

    % Initialize variables to track the best model and its parameters
    bestError = inf;
    bestGam = NaN;
    bestSig = NaN;

    % Initialize a matrix to store gamma, sigma, and RMSE for each combination
    resultsMatrix = [];

    % LSMBM training model using grid search
    for j = 1:length(gam)
        for k = 1:length(sig)
            % Assign the weights
            gamwtr=gam(j) .* w_bpnn;

             % Initialize and train the LS-SVM model
            model = initlssvm(xtr, ytr, 'function estimation', gamwtr, sig(k), 'RBF_kernel');
            model = trainlssvm(model);

            % predictions on the test and training sets
            ytep=simlssvm(model,xte);
            ytrp=simlssvm(model,xtr);

            % Calculate performance evaluation metrics
            errors = postLSIM(ytrp, ytr, ytep, yte);

            % Store Gamma, Sigma, and RMSE in the results matrix
            resultsMatrix = [resultsMatrix; gam(j), sig(k), errors.RMSEte];

            if errors.RMSEte < bestError
                bestError = errors.RMSEte;
                bestModel = model;
                bestGam = gam(j);
                bestSig = sig(k);
            end
        end
    end

    % Record the best hyperparameters for the current fold
    bestParams{i, 1} = ['Fold', num2str(i)];
    bestParams{i, 2} = bestGam;
    bestParams{i, 3} = bestSig;

    % Re-train the best model with optimal parameters
    bestgamwtr = bestGam .* w_bpnn;
    bestModel = initlssvm(xtr, ytr, 'function estimation', bestgamwtr, bestSig, 'RBF_kernel');
    bestModel = trainlssvm(bestModel);

    % Make predictions on the test set
    ytep=simlssvm(bestModel,xte);

    % Make predictions on the training set
    ytrp=simlssvm(bestModel,xtr);

    % Calculate performance evaluation metrics
    finalErrors = postLSIM(ytrp, ytr, ytep, yte);

    % Display evaluation metrics for the best model
    fprintf('Best Gamma: %.2f, Best Sigma: %.2f\n', bestGam, bestSig);
    fprintf('Best RMSE: %.4f\n', finalErrors.RMSEte);                   %display RMSE for the testing set.
    fprintf('Best MAPE: %.4f\n', finalErrors.MAPEte);                   %display MAPE for the testing set.
    fprintf('Best MAE: %.4f\n', finalErrors.MAEte);                        %display MAE for the testing set.
    fprintf('Best R: %.4f\n', finalErrors.Rte);                                  %display R for the testing set
    fprintf('Best R2: %.4f\n', finalErrors.R2te);                              %display R-squared for the testing set.
    disp('---*---');
end
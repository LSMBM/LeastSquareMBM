% POSTLSIM - Calculates performance metrics for regression models.
% This function computes various error metrics both for training and testing data, 
% providing a comprehensive evaluation of model performance.

% Inputs:
%   ytrp - Predicted output values for the training set.
%   ytr  - Actual output values for the training set.
%   ytep - Predicted output values for the testing set.
%   yte  - Actual output values for the testing set.

% Outputs:
%   errors - Struct containing the following error metrics:
%       RMSEte - Root Mean Square Error for the testing set.
%       RMSEtr - Root Mean Square Error for the training set.
%       MAEte  - Mean Absolute Error for the testing set.
%       MAEtr  - Mean Absolute Error for the training set.
%       Rte    - Correlation coefficient for the testing set.
%       Rtr    - Correlation coefficient for the training set.
%       MAPEte - Mean Absolute Percentage Error for the testing set.
%       MAPEtr - Mean Absolute Percentage Error for the training set.
%       R2te   - Coefficient of Determination (R-squared) for the testing set.
%       R2tr   - Coefficient of Determination (R-squared) for the training set.


function errors = postLSIM(ytrp,ytr,ytep,yte)

     % Calculate RMSE for testing and training sets
    errors.RMSEte=sqrt(mse(yte-ytep));
    errors.RMSEtr=sqrt(mse(ytr-ytrp));

    % Calculate MAE for testing and training sets
    errors.MAEte=mean(abs(yte-ytep));
    errors.MAEtr=mean(abs(ytr-ytrp));

    % Calculate MAPE for testing and training sets
    errors.MAPEte=mean(abs((yte-ytep)./yte));
    errors.MAPEtr=mean(abs((ytr-ytrp)./ytr));

    % Calculate correlation coefficients
    correlte=corrcoef(yte,ytep);
    correltr=corrcoef(ytr,ytrp);
    errors.Rte=correlte(2);
    errors.Rtr=correltr(2);

    % Calculate R-squared for testing and training sets
    errors.R2te=correlte(2)^2;
    errors.R2tr=correltr(2)^2;

end
    
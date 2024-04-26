% Normalizer
% This function normalizes a dataset based on the specified type.
% Type 0: No normalization.
% Type 1: Min-max normalization, scaling the features to a range of [0, 1].

function xn = normalizer(x, xmin, xmax, type)
    % x, xmin, xmax, type
    switch type
        case 0
            % Type 0: No normalization applied
            xn = x;
        case 1
            % Type 1: Normalize data
            xn = (x - xmin) ./ (xmax - xmin);
        otherwise
            error('Invalid normalization type specified.');
    end
end

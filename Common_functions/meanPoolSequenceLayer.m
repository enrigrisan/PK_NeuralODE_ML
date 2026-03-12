% FILE: meanPoolSequenceLayer.m
classdef meanPoolSequenceLayer < nnet.layer.Layer
    % Pools a sequence [F x T] -> [F x 1] by mean over T
    methods
        function layer = meanPoolSequenceLayer(varargin)
            layer.Name = 'meanPool';
            if nargin > 0, layer.Name = varargin{1}; end
        end
        function Z = predict(~, X)
            % X is [F x T] (dlarray with 'CT' format)
            Z = mean(X, 2);   % mean over T dimension -> [F x 1]
        end
    end
end
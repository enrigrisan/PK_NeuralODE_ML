% FILE: softplusLayer.m
classdef softplusLayer < nnet.layer.Layer
    methods
        function layer = softplusLayer(varargin)
            layer.Name = 'softplus';
            if nargin > 0, layer.Name = varargin{1}; end
        end
        function Z = predict(~, X)
            Z = log(1 + exp(X));
        end
    end
end
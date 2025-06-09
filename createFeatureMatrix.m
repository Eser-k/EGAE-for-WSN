function X = createFeatureMatrix(Sensors, Model)

%% Developed by Eser Kayali
 % eser-k@web.de

% createFeatureMatrix  Build node feature matrix for GCN input
%
%   X = createFeatureMatrix(Sensors, Model, roundNum) 
%   returns an n×8 matrix X,
%   where each row corresponds to one sensor node (excluding the sink) and
%   each column is a specific feature
%     1) X-coordinate
%     2) Y-coordinate
%     3) Distance to sink
%     4) Node degree (number of neighbors within adjacencyDistance)
%     5) Average distance to neighbors 

    % number of sensor nodes
    n = Model.n;        
    
    % preallocate feature matrix (n rows, 5 columns)
    X = zeros(n, 2);    
    
    for i = 1:n
        % -----------------------------------------------------------------
        % 1–2) Node position
        % -----------------------------------------------------------------
        X(i,1) = Sensors(i).xd; 
        X(i,2) = Sensors(i).yd;
    end
    
end
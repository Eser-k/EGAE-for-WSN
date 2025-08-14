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
    
    % radius for neighbor connectivity
    adjacencyDistance = 30;
    
    % preallocate feature matrix (n rows, 5 columns)
    X = zeros(n, 5);    
    
    for i = 1:n
        % -----------------------------------------------------------------
        % 1–2) Node position
        % -----------------------------------------------------------------
        X(i,1) = Sensors(i).xd; 
        X(i,2) = Sensors(i).yd;
    
        % -----------------------------------------------------------------
        % 3) Distance to sink 
        % ----------------------------------------------------------------- 
        X(i,3) = Sensors(i).dis2sink;
        
        % -----------------------------------------------------------------
        % 4–5) Local neighborhood statistics:
        %       - degree (number of neighbors within adjacencyDistance)
        %       - average neighbor distance
        % -----------------------------------------------------------------
        neighborCount = 0;
        totalNeighborDistance = 0;
    
        for j = 1:n
            if i ~= j
                d = sqrt((Sensors(i).xd - Sensors(j).xd)^2 + (Sensors(i).yd - Sensors(j).yd)^2);
                if d <= adjacencyDistance
                    neighborCount = neighborCount + 1;
                    totalNeighborDistance = totalNeighborDistance + d;
                end
            end
        end
    
        X(i,4) = neighborCount;
    
        if neighborCount > 0
            X(i,5) = (totalNeighborDistance / neighborCount);
        else
            % if no neighbors, use adjacencyDistance as a placeholder
            X(i,5) = adjacencyDistance; 
        end
        
    end
    
end